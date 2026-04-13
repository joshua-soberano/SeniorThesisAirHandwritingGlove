"""
Air Writing Beam Search Decoder  (m5_decoder.py)
-------------------------------------------------
Takes a time series of combined letter + boundary probabilities produced
by the M2 model running online, and decodes them into a word using beam
search with lookahead.

Input (per timestep, streamed online)
---------------------------------------
probs : (T, 30) combined tensor
    columns  0-25 : letter probabilities  (softmax over 26 letters A-Z)
    columns 26-29 : boundary probabilities (softmax over START, INSIDE, END, TRANSITION)

The decoder splits this internally:
    letter_probs   = probs[:, :26]   (T, 26)
    boundary_probs = probs[:, 26:]   (T, 4)

Boundary state indices
----------------------
0 = START
1 = INSIDE
2 = END
3 = TRANSITION

Composite score at emission (Child A)
--------------------------------------
emit_score = alpha * letter_accum[c*]
           + beta  * log(P(END)[t])
           + gamma * log(lm.score(c*, context))

No-emit score (Child B)
------------------------
no_emit_score = beta * log(1 - P(END)[t])

Lookahead
---------
Emission decisions made at timestep t are not pruned until timestep t+L.
This allows L future timesteps of letter and boundary evidence to inform
whether the emission at t was correct before committing to it.

5-gram language model
---------------------
Built from a word list. Uses stupid backoff — falls back to lower order
n-grams with a 0.4 penalty per backoff step. Laplace smoothed.

Output
------
Top decoded word string from the best scoring hypothesis.
Full beam of all B hypotheses with words and scores also available.

Usage
-----
# Standalone file processing mode — accepts combined (T, 30) .npy:
python m5_decoder.py combined_probs.npy --letters A,B,...,Z

# As a module — feed one (30,) vector per timestep:
from m5_decoder import BeamDecoder, NgramLanguageModel
lm      = NgramLanguageModel.from_wordlist('wordlist.txt', letter_classes)
decoder = BeamDecoder(letter_classes, lm)
for t in range(T):
    decoder.step(combined_probs[t])   # (30,) vector
word = decoder.finalise()
"""

import os
import sys
import math
import copy
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

# Boundary state indices — must match M2 training label order
START      = 0
INSIDE     = 1
END        = 2
TRANSITION = 3
BOUNDARY_NAMES = ['START', 'INSIDE', 'END', 'TRANSITION']

# Beam search defaults
BEAM_WIDTH             = 10      # number of hypotheses to keep after pruning
LOOKAHEAD              = 3       # timesteps of future evidence before pruning
MIN_LETTER_INTERVALS   = 5       # minimum timesteps between emissions (250ms at 50ms stride)
TRANSITION_TERMINATION = 10      # sustained TRANSITION timesteps to finalise word

# Composite score weights
ALPHA = 1.0    # letter accumulation weight
BETA  = 0.5    # boundary confidence weight
GAMMA = 0.3    # language model weight

# Language model
NGRAM_ORDER    = 5       # 5-gram
BACKOFF_PENALTY= 0.4     # stupid backoff penalty per order
LAPLACE_K      = 1.0     # Laplace smoothing constant

EPS = 1e-10   # prevent log(0)


# =============================================================================
# LANGUAGE MODEL — KenLM WRAPPER WITH FALLBACK
# =============================================================================

# try to import kenlm — falls back to simple custom LM if not available
try:
    import kenlm as _kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False


class LanguageModel:
    """
    Character-level language model interface.

    Primary  : KenLM binary model trained by m5_build_lm.py on WikiText-103.
    Fallback : Simple count-based n-gram with stupid backoff, built from a
               word list or uniform if no word list provided.

    Both expose the same interface:
        lm.log_score(char, context_tuple) → float (log probability)

    KenLM format note
    -----------------
    KenLM was trained on space-separated characters e.g. "T H E".
    At query time we convert the context tuple ('T','H') to "T H"
    and query the next character 'E' giving score for "T H E".
    KenLM returns log10 probabilities — we convert to natural log.
    """

    def __init__(self):
        self._kenlm_model  = None     # KenLM model object
        self._counts       = None     # fallback count dicts
        self._letter_classes = None
        self._n_classes    = 0
        self._order        = NGRAM_ORDER
        self._mode         = 'uniform'

    # ------------------------------------------------------------------
    # Pickle model path (pure Python, no kenlm needed)
    # ------------------------------------------------------------------

    @classmethod
    def from_pkl(cls, pkl_path: str,
                 letter_classes: list) -> 'LanguageModel':
        """
        Load a pure Python n-gram model built by m5_build_lm.py.
        No kenlm installation needed — uses pickle log_probs directly.
        """
        import pickle
        lm = cls()
        lm._letter_classes = letter_classes
        lm._n_classes      = len(letter_classes)

        if not os.path.isfile(pkl_path):
            print(f"  WARNING: pkl model not found at {pkl_path} — using uniform LM.")
            lm._mode = 'uniform'
            return lm

        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)

        lm._pkl_log_probs = model['log_probs']
        lm._order         = model['order']
        lm._mode          = 'pkl'
        print(f"  Pkl LM loaded: {pkl_path}")
        print(f"  Order: {lm._order}-gram  "
              f"Vocab: {len(model['vocab'])} chars  "
              f"Words trained on: WikiText-103")
        return lm

    def _pkl_log_score(self, char: str, context: tuple) -> float:
        """Score using pure Python pickle model."""
        for n in range(min(self._order - 1, len(context)), -1, -1):
            ctx = context[-n:] if n > 0 else ()
            ctx_data = self._pkl_log_probs.get(n + 1, {}).get(ctx, {})
            if ctx_data and char in ctx_data:
                return ctx_data[char] * math.log(10)  # log10 → natural log
        return math.log(1.0 / max(self._n_classes, 1))

    # ------------------------------------------------------------------
    # KenLM path
    # ------------------------------------------------------------------

    @classmethod
    def from_binary(cls, binary_path: str,
                    letter_classes: list) -> 'LanguageModel':
        """
        Load a KenLM binary model built by m5_build_lm.py.
        Falls back to uniform model if KenLM not installed.
        """
        lm = cls()
        lm._letter_classes = letter_classes
        lm._n_classes      = len(letter_classes)

        if not KENLM_AVAILABLE:
            print("  WARNING: kenlm not installed — using uniform LM.")
            print("  Install with: pip install kenlm")
            lm._mode = 'uniform'
            return lm

        if not os.path.isfile(binary_path):
            print(f"  WARNING: KenLM binary not found at {binary_path} — using uniform LM.")
            lm._mode = 'uniform'
            return lm

        lm._kenlm_model = _kenlm.Model(binary_path)
        lm._mode        = 'kenlm'
        lm._order       = lm._kenlm_model.order
        print(f"  KenLM model loaded: {binary_path}")
        print(f"  Order: {lm._order}-gram")
        return lm

    # ------------------------------------------------------------------
    # Fallback: count-based path
    # ------------------------------------------------------------------

    @classmethod
    def from_wordlist(cls, path: str,
                      letter_classes: list,
                      order: int = NGRAM_ORDER) -> 'LanguageModel':
        """
        Build fallback count-based n-gram from a plain text word list.
        Used when KenLM binary is not available.
        """
        lm = cls()
        lm._letter_classes = letter_classes
        lm._n_classes      = len(letter_classes)
        lm._order          = order
        lm._mode           = 'counts'
        lm._counts         = [{} for _ in range(order + 1)]
        valid              = set(letter_classes)
        total              = 0

        with open(path, 'r') as f:
            for line in f:
                word = line.strip().upper()
                word = ''.join(c for c in word if c in valid)
                if not word:
                    continue
                padded = ('_',) * (order - 1) + tuple(word)
                for i in range(len(padded) - order + 1):
                    for n in range(1, order + 1):
                        ctx  = padded[i + order - n: i + order - 1]
                        char = padded[i + order - 1]
                        if ctx not in lm._counts[n]:
                            lm._counts[n][ctx] = defaultdict(float)
                        lm._counts[n][ctx][char] += 1.0
                total += len(word)

        print(f"  Fallback LM built from {path} ({total:,} chars, order={order})")
        return lm

    @classmethod
    def uniform(cls, letter_classes: list) -> 'LanguageModel':
        """Uniform fallback — all letters equally likely."""
        lm = cls()
        lm._letter_classes = letter_classes
        lm._n_classes      = len(letter_classes)
        lm._order          = 1
        lm._mode           = 'uniform'
        print("  Using uniform language model.")
        return lm

    # ------------------------------------------------------------------
    # Shared scoring interface
    # ------------------------------------------------------------------

    @property
    def lm_order(self) -> int:
        """Order of the language model — used by Hypothesis context window."""
        return self._order

    def log_score(self, char: str, context: tuple) -> float:
        """
        Return log P(char | context).
        context : tuple of preceding characters e.g. ('T', 'H')
        Returns natural log probability (always finite — never -inf).
        """
        if self._mode == 'kenlm':
            return self._kenlm_log_score(char, context)
        elif self._mode == 'pkl':
            return self._pkl_log_score(char, context)
        elif self._mode == 'counts':
            return self._counts_log_score(char, context)
        else:
            return math.log(1.0 / max(self._n_classes, 1))

    def _kenlm_log_score(self, char: str, context: tuple) -> float:
        """
        Query KenLM for P(char | context).
        KenLM was trained on space-separated chars e.g. "T H E".
        We build the query string from context + char and score it.
        KenLM returns log10 — convert to natural log.
        """
        # build space-separated string of context chars + target char
        # KenLM's score() method takes the full sequence and returns
        # the sum of log10 probs — we only want the last token's prob
        # so we use full_scores() to get per-token scores
        ctx_chars  = list(context) + [char]
        query      = ' '.join(ctx_chars)
        # full_scores yields (log10_prob, ngram_length, oov) per token
        scores     = list(self._kenlm_model.full_scores(query,
                                                         bos=False, eos=False))
        if not scores:
            return math.log(1.0 / max(self._n_classes, 1))
        log10_prob = scores[-1][0]                  # last token = char
        return max(log10_prob * math.log(10), math.log(EPS))

    def _counts_log_score(self, char: str, context: tuple) -> float:
        """Stupid backoff count-based scoring with Laplace smoothing."""
        penalty = 1.0
        for n in range(min(len(context) + 1, self._order), 0, -1):
            ctx = context[-(n-1):] if n > 1 else ()
            if ctx in self._counts[n]:
                ctx_counts = self._counts[n][ctx]
                ctx_total  = sum(ctx_counts.values())
                num = ctx_counts.get(char, 0.0) + LAPLACE_K
                den = ctx_total + LAPLACE_K * self._n_classes
                return math.log(max(penalty * num / den, EPS))
            penalty *= BACKOFF_PENALTY
        return math.log(max(penalty / max(self._n_classes, 1), EPS))


# =============================================================================
# HYPOTHESIS
# =============================================================================

class Hypothesis:
    """
    A single beam hypothesis representing one interpretation of the stream.

    Attributes
    ----------
    word         : letters emitted so far
    score        : cumulative log composite score
    letter_accum : {letter: float} log-prob accumulated since last reset
    context      : tuple of last (order-1) emitted chars for LM lookup
    last_emit_t  : timestep of last emission
    created_t    : timestep this hypothesis was born (for buffer management)
    """

    def __init__(self, letter_classes: list, lm_order: int,
                 created_t: int = 0):
        self.word         = ''
        self.score        = 0.0
        self.letter_accum = {c: 0.0 for c in letter_classes}
        self.accum_count  = 0        # number of timesteps accumulated since last reset
        self.context      = ('_',) * (lm_order - 1)   # start tokens
        self.last_emit_t  = -MIN_LETTER_INTERVALS      # allow immediate emission at t=0
        self.created_t    = created_t
        self.letter_classes = letter_classes
        self.lm_order     = lm_order

    def copy(self) -> 'Hypothesis':
        h               = Hypothesis.__new__(Hypothesis)
        h.word          = self.word
        h.score         = self.score
        h.letter_accum  = self.letter_accum.copy()
        h.accum_count   = self.accum_count
        h.context       = self.context
        h.last_emit_t   = self.last_emit_t
        h.created_t     = self.created_t
        h.letter_classes= self.letter_classes
        h.lm_order      = self.lm_order
        return h

    def accumulate(self, letter_probs: np.ndarray):
        """Add log letter probabilities to accumulator. Always called every timestep."""
        for i, c in enumerate(self.letter_classes):
            self.letter_accum[c] += math.log(max(float(letter_probs[i]), EPS))
        self.accum_count += 1

    def best_emit_letter(self, boundary_probs: np.ndarray,
                         lm: LanguageModel) -> tuple:
        """
        Find the best letter to emit and its composite score.
        letter_accum is normalised by accum_count so it is a per-timestep
        average — keeps it comparable in scale to the boundary and LM terms.

        A baseline offset is subtracted so that a completely uncertain emission
        (uniform letter distribution, low P(END)) scores near zero — the same
        as a no-emit. A confident emission (peaked letter distribution, high
        P(END)) scores positively and wins over no-emit. An uncertain emission
        scores negatively and loses to no-emit.

        A timing prior rewards emissions at natural inter-letter durations
        (1-3 seconds = 20-60 intervals at 50ms stride) and penalises
        rapid back-to-back emissions (< 10 intervals = 500ms).

        baseline = ALPHA * log(1/n_letters) + BETA * log(0.5)
        Returns (letter, composite_score - baseline + timing_prior).
        """
        log_p_end    = math.log(max(float(boundary_probs[END]), EPS))
        n            = max(1, self.accum_count)
        n_letters    = len(self.letter_classes)
        baseline     = ALPHA * math.log(1.0 / n_letters) + BETA * math.log(0.5)

        # timing prior — based on intervals since last emission
        # interval = 50ms, so 20 intervals = 1s, 60 intervals = 3s
        intervals_since = n   # accum_count == intervals since last emit/reset
        TIMING_WEIGHT   = 0.3  # keep lower than boundary weight to not dominate
        PEAK_INTERVALS  = 30   # peak reward at 30 intervals = 1.5s
        MIN_INTERVALS   = 10   # heavy penalty below 10 intervals = 500ms
        if intervals_since < MIN_INTERVALS:
            # steep penalty for very fast emissions
            timing_prior = TIMING_WEIGHT * math.log(
                max(intervals_since / MIN_INTERVALS, 1e-6))
        else:
            # gaussian-shaped reward peaking at PEAK_INTERVALS, decaying beyond
            sigma        = 20.0   # spread in intervals (~1 second)
            timing_prior = TIMING_WEIGHT * math.exp(
                -0.5 * ((intervals_since - PEAK_INTERVALS) / sigma) ** 2)

        best_letter  = None
        best_score   = -math.inf

        for c in self.letter_classes:
            normalised_accum = self.letter_accum[c] / n
            composite = (ALPHA * normalised_accum
                       + BETA  * log_p_end
                       + GAMMA * lm.log_score(c, self.context))
            if composite > best_score:
                best_score  = composite
                best_letter = c

        return best_letter, best_score - baseline + timing_prior

    def emit(self, letter: str, emit_score: float, t: int) -> 'Hypothesis':
        """
        Return a new hypothesis with letter emitted and accumulator reset.
        Does not modify self.
        """
        h               = self.copy()
        h.word          = self.word + letter
        h.score        += emit_score
        h.letter_accum  = {c: 0.0 for c in self.letter_classes}   # reset
        h.accum_count   = 0                                         # reset count
        h.context       = self.context[1:] + (letter,)             # slide window
        h.last_emit_t   = t
        h.created_t     = t
        return h

    def no_emit(self, t: int) -> 'Hypothesis':
        """
        Return a new hypothesis without emitting. No score penalty —
        the hypothesis simply carries forward unchanged.
        Not emitting is the default; the emit score alone discriminates.
        """
        h           = self.copy()
        h.created_t = t
        return h

    def can_emit(self, t: int) -> bool:
        """True if minimum letter duration constraint is satisfied."""
        return (t - self.last_emit_t) >= MIN_LETTER_INTERVALS

    def __repr__(self):
        return (f"Hypothesis(word='{self.word}', score={self.score:.3f}, "
                f"last_emit={self.last_emit_t})")


# =============================================================================
# BEAM DECODER
# =============================================================================

class BeamDecoder:
    """
    Letter-level beam search decoder for the boundary-aware model.

    Rather than branching at every timestep, the decoder:
      1. Accumulates letter evidence continuously across all timesteps
      2. Detects END windows using the same logic as the greedy decoder
      3. Only branches the beam at END window falling edges — once per letter
      4. At each emission point takes the top-K letters and creates K new
         hypotheses, pruning back to beam_width by composite score

    This is vastly faster than timestep-level beam search and avoids the
    emit/no-emit competition problem that caused the original beam to never emit.
    """

    def __init__(self, letter_classes: list,
                 lm: LanguageModel,
                 beam_width: int            = BEAM_WIDTH,
                 lookahead: int             = LOOKAHEAD,
                 min_letter_intervals: int  = MIN_LETTER_INTERVALS,
                 transition_termination: int= TRANSITION_TERMINATION,
                 alpha: float               = ALPHA,
                 beta: float                = BETA,
                 gamma: float               = GAMMA,
                 end_threshold: float       = 0.25,
                 debug: bool                = False):

        self.letter_classes          = letter_classes
        self.lm                      = lm
        self.beam_width              = beam_width
        self.min_letter_intervals    = min_letter_intervals
        self.transition_termination  = transition_termination
        self.end_threshold           = end_threshold
        self.debug                   = debug

        global ALPHA, BETA, GAMMA
        ALPHA, BETA, GAMMA = alpha, beta, gamma

        self.t = 0

        # each hypothesis: (word_str, score, lm_context, last_emit_t, accum, accum_count)
        lm_order   = lm.lm_order if hasattr(lm, 'lm_order') else 1
        init_ctx   = ('_',) * max(0, lm_order - 1)
        self.beam  = [{'word': '', 'score': 0.0, 'ctx': init_ctx,
                        'last_emit_t': -min_letter_intervals,
                        'accum': np.zeros(len(letter_classes), dtype=np.float32),
                        'accum_count': 0}]

        # window state — same as greedy decoder
        self.in_end_window   = False
        self.consecutive_end = 0
        self.min_window_len  = 2
        self.best_ltr_conf   = 0.0
        self.best_ltr_t      = 0
        self.best_ltr_probs  = None   # letter prob vector at best confidence timestep

        self.transition_counter = 0
        self.history = []

        if self.debug:
            print(f"\n  [BeamDecoder letter-level] beam_width={beam_width} "
                  f"end_threshold={end_threshold} alpha={ALPHA} beta={BETA} gamma={GAMMA}")

    def step(self, probs: np.ndarray) -> list:
        n_letters      = len(self.letter_classes)
        letter_probs   = probs[:n_letters].astype(np.float32)
        boundary_probs = probs[n_letters:].astype(np.float32)
        t              = self.t
        p_end          = float(boundary_probs[END])

        # accumulate letter evidence into all hypotheses
        for h in self.beam:
            h['accum']       += np.log(np.maximum(letter_probs, 1e-10))
            h['accum_count'] += 1

        # track best letter confidence within END window
        ltr_conf = float(letter_probs.max())
        if p_end >= self.end_threshold:
            self.consecutive_end += 1
            if self.consecutive_end >= self.min_window_len:
                self.in_end_window = True
            if ltr_conf > self.best_ltr_conf:
                self.best_ltr_conf  = ltr_conf
                self.best_ltr_t     = t
                self.best_ltr_probs = letter_probs.copy()
        else:
            # falling edge — emit if window was confirmed
            if self.in_end_window and self.best_ltr_probs is not None:
                self._emit_letter(self.best_ltr_probs, self.best_ltr_t)

            self.in_end_window   = False
            self.consecutive_end = 0
            self.best_ltr_conf   = 0.0
            self.best_ltr_probs  = None

        # transition termination
        dominant_state = int(np.argmax(boundary_probs))
        if dominant_state == TRANSITION:
            self.transition_counter += 1
        else:
            self.transition_counter = 0

        self.t += 1
        return self.beam

    def _emit_letter(self, letter_probs: np.ndarray, t: int):
        """
        Branch the beam at an END window falling edge.
        For each existing hypothesis, create beam_width children using the
        top-K letters by composite score (acoustic + LM + timing prior).
        Prune back to beam_width.
        """
        n_letters = len(self.letter_classes)
        new_beam  = []

        for h in self.beam:
            # skip if cooldown not satisfied
            if (t - h['last_emit_t']) < self.min_letter_intervals:
                new_beam.append(h)
                continue

            n           = max(1, h['accum_count'])
            norm_accum  = h['accum'] / n   # per-timestep average log prob

            # baseline for scoring
            n_letters_count = len(self.letter_classes)
            baseline = ALPHA * math.log(1.0 / n_letters_count) + BETA * math.log(0.5)

            # timing prior
            intervals_since = n
            PEAK_INTERVALS  = 30
            MIN_INTERVALS   = 10
            if intervals_since < MIN_INTERVALS:
                timing = 0.3 * math.log(max(intervals_since / MIN_INTERVALS, 1e-6))
            else:
                sigma  = 20.0
                timing = 0.3 * math.exp(-0.5 * ((intervals_since - PEAK_INTERVALS) / sigma) ** 2)

            # score each letter
            letter_scores = []
            for i, letter in enumerate(self.letter_classes):
                ctx      = h['ctx']
                lm_score = GAMMA * self.lm.log_score(letter, ctx)
                composite = (ALPHA * float(norm_accum[i])
                             + BETA  * math.log(max(float(letter_probs[i]), 1e-10))
                             + lm_score)
                emit_score = composite - baseline + timing
                letter_scores.append((emit_score, letter, i))

            # take top beam_width letters
            letter_scores.sort(reverse=True)
            top_k = letter_scores[:self.beam_width]

            lm_order = self.lm.lm_order if hasattr(self.lm, 'lm_order') else 1

            for emit_score, letter, li in top_k:
                new_ctx = (h['ctx'] + (letter,))[-(lm_order-1):] \
                          if lm_order > 1 else ()
                new_beam.append({
                    'word'        : h['word'] + letter,
                    'score'       : h['score'] + emit_score,
                    'ctx'         : new_ctx,
                    'last_emit_t' : t,
                    'accum'       : np.zeros(n_letters, dtype=np.float32),
                    'accum_count' : 0,
                })

            if self.debug:
                best_letter, best_score = letter_scores[0][1], letter_scores[0][0]
                print(f'    t={t:>3}  EMIT "{best_letter}"  '
                      f'score={best_score:.3f}  word="{h["word"]}"')

        # prune to beam_width
        new_beam.sort(key=lambda h: h['score'], reverse=True)
        self.beam = new_beam[:self.beam_width]

    def finalise(self) -> str:
        """Return top hypothesis word."""
        # flush any open END window at sequence end
        if self.in_end_window and self.best_ltr_probs is not None:
            self._emit_letter(self.best_ltr_probs, self.best_ltr_t)

        if not self.beam:
            return ''
        return self.beam[0]['word']

    @property
    def should_terminate(self) -> bool:
        return self.transition_counter >= TRANSITION_TERMINATION

    @property
    def top_word(self) -> str:
        return self.beam[0]['word'] if self.beam else ''

    @property
    def top_score(self) -> float:
        return self.beam[0]['score'] if self.beam else float('-inf')

    def beam_summary(self) -> list:
        return [(h['word'], round(h['score'], 4)) for h in self.beam]


# =============================================================================
# STANDALONE FILE PROCESSING
# =============================================================================

def load_probs(combined_path: str, n_letters: int = 26):
    """
    Load combined probability array from a .npy file.
    combined_path : (T, n_letters + 4) array
        columns  0 to n_letters-1 : letter probabilities
        columns  n_letters to end : boundary probabilities (4 columns)
    """
    probs = np.load(combined_path).astype(np.float32)

    assert probs.ndim == 2, \
        f"Expected (T, {n_letters + 4}) array, got shape {probs.shape}"
    assert probs.shape[1] == n_letters + 4, \
        f"Expected {n_letters + 4} columns (letters + boundary), got {probs.shape[1]}"

    return probs


def run_file(combined_path: str,
             letter_classes: list,
             lm_binary_path: str    = None,
             wordlist_path: str     = None,
             beam_width: int        = BEAM_WIDTH,
             lookahead: int         = LOOKAHEAD,
             alpha: float           = ALPHA,
             beta: float            = BETA,
             gamma: float           = GAMMA,
             verbose: bool          = False) -> str:
    """
    Run beam search decoder on a combined (T, 30) probability array.
    Returns decoded word.

    Language model priority:
        1. KenLM binary  (lm_binary_path) — best, from m5_build_lm.py
        2. Word list     (wordlist_path)  — fallback count-based LM
        3. Uniform                        — last resort
    """
    print("=" * 60)
    print("Air Writing Beam Search Decoder  (m5_decoder.py)")
    print("=" * 60)

    # load combined probs
    probs     = load_probs(combined_path, n_letters=len(letter_classes))
    T         = probs.shape[0]
    print(f"\nLoaded: {T} timesteps")
    print(f"  Letter columns   : {len(letter_classes)} ({letter_classes[0]}..{letter_classes[-1]})")
    print(f"  Boundary columns : 4 (START, INSIDE, END, TRANSITION)")
    print(f"Letter classes: {letter_classes}")

    # build language model — KenLM preferred
    if lm_binary_path:
        lm = LanguageModel.from_binary(lm_binary_path, letter_classes)
    elif wordlist_path and os.path.isfile(wordlist_path):
        lm = LanguageModel.from_wordlist(wordlist_path, letter_classes)
    else:
        lm = LanguageModel.uniform(letter_classes)

    # build decoder
    decoder = BeamDecoder(
        letter_classes          = letter_classes,
        lm                      = lm,
        beam_width              = beam_width,
        lookahead               = lookahead,
        alpha                   = alpha,
        beta                    = beta,
        gamma                   = gamma,
    )

    print(f"\nDecoding {T} timesteps...")
    print(f"  Beam width : {beam_width}")
    print(f"  Lookahead  : {lookahead} timesteps ({lookahead * 50}ms)")
    print(f"  Alpha/Beta/Gamma: {alpha}/{beta}/{gamma}")

    if verbose:
        n_ltr = len(letter_classes)
        print(f"\n  {'t':>5}  {'Dom State':>12}  {'P(END)':>8}  "
              f"{'Top Letter':>12}  {'Beam Top':>15}  {'Score':>8}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*15}  {'-'*8}")

    # stream through all timesteps
    for t in range(T):
        confirmed = decoder.step(probs[t])

        if verbose:
            n_ltr      = len(letter_classes)
            bp         = probs[t, n_ltr:]
            lp         = probs[t, :n_ltr]
            dom_state  = BOUNDARY_NAMES[int(np.argmax(bp))]
            p_end      = float(bp[END])
            top_letter = letter_classes[int(np.argmax(lp))]
            top_word   = decoder.top_word
            top_score  = decoder.top_score
            print(f"  {t:>5}  {dom_state:>12}  {p_end:>8.3f}  "
                  f"{top_letter:>12}  {top_word:>15}  {top_score:>8.3f}")

        if decoder.should_terminate:
            print(f"\n  Termination: sustained TRANSITION at t={t}")
            break

    # finalise
    word = decoder.finalise()

    # results
    print("\n" + "=" * 60)
    print("DECODING RESULTS")
    print("=" * 60)
    print(f"\nDecoded word : {word}")
    print(f"\nFull beam (top {beam_width} hypotheses):")
    print(f"  {'Rank':>4}  {'Word':>15}  {'Score':>10}")
    print(f"  {'-'*4}  {'-'*15}  {'-'*10}")
    for rank, (w, s) in enumerate(decoder.beam_summary(), 1):
        print(f"  {rank:>4}  {w:>15}  {s:>10.4f}")
    print("=" * 60)

    return word


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Air Writing Beam Search Decoder'
    )
    parser.add_argument('combined_probs',
                        help='Path to (T, n_letters+4) combined .npy file')
    parser.add_argument('--letters',
                        help='Comma-separated letter classes e.g. A,B,C,...,Z',
                        default=','.join([chr(i) for i in range(65, 91)]))
    parser.add_argument('--lm',
                        help='Path to KenLM binary model from m5_build_lm.py '
                             '(e.g. lm/wiki103_char_5gram.binary)',
                        default=None)
    parser.add_argument('--wordlist',
                        help='Fallback: path to word list for count-based LM',
                        default=None)
    parser.add_argument('--beam',      type=int,   default=BEAM_WIDTH)
    parser.add_argument('--lookahead', type=int,   default=LOOKAHEAD)
    parser.add_argument('--alpha',     type=float, default=ALPHA)
    parser.add_argument('--beta',      type=float, default=BETA)
    parser.add_argument('--gamma',     type=float, default=GAMMA)
    parser.add_argument('--verbose',   action='store_true')

    if len(sys.argv) == 1:
        # interactive mode
        combined_path  = input("Path to combined probs .npy file (T, 30): ").strip()
        letters_str    = input("Letter classes (comma-separated, default A-Z): ").strip()
        letter_classes = [l.strip().upper() for l in letters_str.split(',')] \
                         if letters_str else [chr(i) for i in range(65, 91)]
        lm_path        = input("Path to KenLM binary .binary (or blank): ").strip() or None
        wordlist_path  = input("Path to fallback wordlist (or blank): ").strip() or None
        verbose_str    = input("Verbose output? (y/n): ").strip().lower()
        verbose        = verbose_str == 'y'

        run_file(combined_path, letter_classes,
                 lm_binary_path=lm_path,
                 wordlist_path=wordlist_path,
                 verbose=verbose)
    else:
        args = parser.parse_args()
        letter_classes = [l.strip().upper() for l in args.letters.split(',')]
        run_file(
            args.combined_probs,
            letter_classes,
            lm_binary_path = args.lm,
            wordlist_path  = args.wordlist,
            beam_width     = args.beam,
            lookahead      = args.lookahead,
            alpha          = args.alpha,
            beta           = args.beta,
            gamma          = args.gamma,
            verbose        = args.verbose,
        )


if __name__ == '__main__':
    main()
