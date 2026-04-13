"""
Air Writing Language Model Builder  (m5_build_lm.py)
------------------------------------------------------
Downloads WikiText-103, extracts character-level text,
and trains a 5-gram KenLM language model on it.

The resulting .arpa and .binary files are used by m5_decoder.py
for character-level language model scoring during beam search.

Dependencies
------------
pip install kenlm datasets
KenLM binaries must be on PATH:
    sudo apt-get install build-essential cmake
    git clone https://github.com/kpu/kenlm
    cd kenlm && mkdir -p build && cd build && cmake .. && make -j4
    export PATH=$PATH:/path/to/kenlm/build/bin

Output files
------------
wiki103_char_5gram.arpa    — ARPA format LM (human readable)
wiki103_char_5gram.binary  — KenLM binary format (fast loading)
wiki103_char_corpus.txt    — preprocessed character corpus

Usage
-----
python m5_build_lm.py
python m5_build_lm.py --order 5 --out_dir ./lm --max_tokens 10000000
"""

import os
import sys
import re
import argparse
import subprocess
import tempfile
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ORDER    = 5
DEFAULT_OUT_DIR  = './lm'
DEFAULT_MAX_TOKENS = 10_000_000   # cap corpus size for speed
VALID_CHARS      = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
SPACE_TOKEN      = '<SPACE>'      # represents word boundary in char LM


# =============================================================================
# STEP 1 — DOWNLOAD WIKITEXT-103
# =============================================================================

def download_wikitext103(max_tokens: int) -> list:
    """
    Download WikiText-103 train split using HuggingFace datasets.
    Returns list of cleaned uppercase word strings.
    """
    print("Downloading WikiText-103 train split...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    dataset = load_dataset('wikitext', 'wikitext-103-v1',
                           split='train', trust_remote_code=True)

    words = []
    total_chars = 0
    for item in dataset:
        text = item['text']
        if not text.strip():
            continue
        # extract only alphabetic words, uppercase
        for word in re.findall(r"[A-Za-z]+", text):
            w = word.upper()
            words.append(w)
            total_chars += len(w)
            if total_chars >= max_tokens:
                print(f"  Reached {max_tokens:,} token cap at {len(words):,} words.")
                return words

    print(f"  Loaded {len(words):,} words ({total_chars:,} characters).")
    return words


# =============================================================================
# STEP 2 — BUILD CHARACTER CORPUS
# =============================================================================

def build_char_corpus(words: list, corpus_path: str):
    """
    Write character-level corpus to file.
    Each word is written as space-separated characters.
    Word boundaries are represented by a newline (treated as sentence boundary
    by KenLM — this gives the LM context of surrounding characters within a word
    and a natural boundary between words).

    Format per line:
        T H E
        Q U I C K
        B R O W N
        ...

    This means KenLM trains on character sequences within words,
    with word boundaries as sentence boundaries.
    """
    print(f"Writing character corpus to: {corpus_path}")
    total_lines = 0
    with open(corpus_path, 'w') as f:
        for word in words:
            # filter to valid uppercase letters only
            chars = [c for c in word if c in VALID_CHARS]
            if len(chars) == 0:
                continue
            f.write(' '.join(chars) + '\n')
            total_lines += 1

    print(f"  Written {total_lines:,} word lines ({sum(len(w) for w in words):,} characters).")
    return corpus_path


# =============================================================================
# STEP 3 — TRAIN KENLM
# =============================================================================

def train_kenlm(corpus_path: str, order: int,
                arpa_path: str, binary_path: str):
    """
    Run KenLM lmplz to build ARPA model, then build_binary for fast loading.
    Requires kenlm binaries on PATH.
    """
    # check lmplz is available
    try:
        result = subprocess.run(['lmplz', '--help'],
                                capture_output=True, text=True)
    except FileNotFoundError:
        print("ERROR: lmplz not found on PATH.")
        print("Install KenLM:")
        print("  git clone https://github.com/kpu/kenlm")
        print("  cd kenlm && mkdir -p build && cd build && cmake .. && make -j4")
        print("  export PATH=$PATH:$(pwd)/bin")
        sys.exit(1)

    # train ARPA
    print(f"\nTraining {order}-gram KenLM on {corpus_path}...")
    print(f"  Output ARPA: {arpa_path}")
    with open(corpus_path, 'r') as corpus_f, \
         open(arpa_path, 'w') as arpa_f:
        proc = subprocess.run(
            ['lmplz',
             '-o', str(order),
             '--discount_fallback',    # handles low-count n-grams gracefully
             '--prune', '0', '0', '1', # prune trigrams and above with count < 1
             ],
            stdin  = corpus_f,
            stdout = arpa_f,
            stderr = subprocess.PIPE,
            text   = True
        )
    if proc.returncode != 0:
        print(f"ERROR: lmplz failed:\n{proc.stderr}")
        sys.exit(1)
    print(f"  ARPA model written: {arpa_path}")

    # convert to binary
    print(f"\nConverting to binary: {binary_path}")
    proc = subprocess.run(
        ['build_binary', 'trie', arpa_path, binary_path],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"ERROR: build_binary failed:\n{proc.stderr}")
        sys.exit(1)
    print(f"  Binary model written: {binary_path}")

    # report sizes
    arpa_size   = os.path.getsize(arpa_path)   / 1e6
    binary_size = os.path.getsize(binary_path) / 1e6
    print(f"\n  ARPA   : {arpa_size:.1f} MB")
    print(f"  Binary : {binary_size:.1f} MB")


# =============================================================================
# STEP 4 — VERIFY
# =============================================================================

def verify_lm(binary_path: str, letter_classes: list):
    """
    Quick sanity check — score a few common character sequences.
    """
    try:
        import kenlm
    except ImportError:
        print("\nSkipping verification — kenlm Python bindings not installed.")
        print("Install with: pip install kenlm")
        return

    print(f"\nVerifying: {binary_path}")
    model = kenlm.Model(binary_path)

    test_sequences = [
        'T H E',       # very common
        'I N G',       # common suffix
        'Q Z X',       # rare sequence
        'A B C',       # arbitrary
    ]

    print(f"  {'Sequence':>10}  {'Log10 Prob':>12}")
    print(f"  {'-'*10}  {'-'*12}")
    for seq in test_sequences:
        score = model.score(seq, bos=True, eos=True)
        print(f"  {seq:>10}  {score:>12.4f}")

    print("\nVerification complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build character-level KenLM from WikiText-103'
    )
    parser.add_argument('--order',      type=int, default=DEFAULT_ORDER,
                        help=f'N-gram order (default {DEFAULT_ORDER})')
    parser.add_argument('--out_dir',    type=str, default=DEFAULT_OUT_DIR,
                        help=f'Output directory (default {DEFAULT_OUT_DIR})')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'Max characters to use (default {DEFAULT_MAX_TOKENS:,})')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download, use existing corpus file')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    corpus_path = os.path.join(args.out_dir, 'wiki103_char_corpus.txt')
    arpa_path   = os.path.join(args.out_dir,
                               f'wiki103_char_{args.order}gram.arpa')
    binary_path = os.path.join(args.out_dir,
                               f'wiki103_char_{args.order}gram.binary')

    print("=" * 60)
    print("Air Writing Language Model Builder  (m5_build_lm.py)")
    print("=" * 60)
    print(f"\n  N-gram order : {args.order}")
    print(f"  Output dir   : {args.out_dir}")
    print(f"  Max tokens   : {args.max_tokens:,}")

    # step 1 — download
    if not args.skip_download:
        words = download_wikitext103(args.max_tokens)
    else:
        print(f"\nSkipping download — using existing corpus: {corpus_path}")
        if not os.path.isfile(corpus_path):
            print(f"ERROR: corpus not found at {corpus_path}")
            sys.exit(1)
        words = None

    # step 2 — build corpus
    if words is not None:
        build_char_corpus(words, corpus_path)

    # step 3 — train KenLM
    train_kenlm(corpus_path, args.order, arpa_path, binary_path)

    # step 4 — verify
    letter_classes = [chr(i) for i in range(65, 91)]
    verify_lm(binary_path, letter_classes)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  ARPA   : {arpa_path}")
    print(f"  Binary : {binary_path}")
    print(f"\nTo use in m5_decoder.py:")
    print(f"  python m5_decoder.py combined_probs.npy \\")
    print(f"      --lm {binary_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
