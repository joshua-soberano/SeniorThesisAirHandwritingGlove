"""
Air Writing Preprocessing Pipeline  (m1_preprocess.py)
-------------------------------------------------------
Input  : .txt file produced by m1_record_and_label.py
Output : _intervals.txt file with columns IntervalStart, IntervalEnd, Label

Pipeline
--------
1. Load file, skip % comment lines
2. Global z-score normalization of 4 sensor channels
3. Extract 10 random 100ms rest snippets from [1s, first_button_press) -> label A
4. For each B/C/D/E button press:
   a. Compute 300ms pre-click baseline per channel
   b. Build smoothed multi-channel activity signal A(t)
   c. Detect onset  via hysteresis T_on  = mu_A + 4*sigma_A  (min 40ms continuous)
   d. Detect offset via hysteresis T_off = mu_A + 2*sigma_A  (min 75ms continuous)
   e. Slide 100ms windows with 50ms stride over [onset, offset]
5. Write IntervalStart, IntervalEnd, Label to output file
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

CHANNEL_COLS   = ['Channel1', 'Channel2', 'Channel3', 'Channel4']
CHANNEL_NAMES  = ['stretch_1', 'stretch_2', 'flex_1', 'flex_2']   # internal names

SMOOTH_MS       = 30      # moving-average window for activity signal (ms)
BASELINE_MS     = 300     # pre-click baseline window (ms)
SKIP_MS         = 100     # ignore this many ms after click (click disturbance)
ONSET_SEARCH_MS = 500     # max ms after click to search for onset
TON_SIGMA       = 4.0     # T_on  = mu_A + TON_SIGMA * sigma_A
TOFF_ALPHA      = 0.15    # T_off = TOFF_ALPHA * M_peak (90th percentile of motion)
PEAK_PERCENTILE = 90      # percentile of post-onset activity used as M_peak
ONSET_MIN_MS    = 40      # onset must exceed T_on continuously for this long
OFFSET_MIN_MS   = 75      # offset must stay below T_off continuously for this long
MIN_MOTION_MS   = 80      # reject trial if motion duration shorter than this
MAX_MOTION_MS   = 3000    # reject trial if motion duration longer than this
WINDOW_MS       = 100     # sliding window size
STRIDE_MS       = 50      # sliding window stride
REST_SKIP_SEC   = 1.0     # skip first second of file for rest sampling
REST_N          = 10      # number of rest snippets to extract
REST_WIN_MS     = 100     # rest snippet duration (ms)
EPS             = 1e-8    # small value to avoid division by zero


# =============================================================================
# HELPERS
# =============================================================================

def ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average — output same length as input."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    # Pad left so output is same length and causal
    padded = np.concatenate([np.full(window - 1, x[0]), x])
    return np.convolve(padded, kernel, mode='valid')


def continuous_threshold(signal: np.ndarray,
                         condition: np.ndarray,
                         min_samples: int) -> Optional[int]:
    """
    Return index of first run in `condition` (bool array) that lasts at
    least `min_samples` consecutive True values.  Returns None if not found.
    """
    count = 0
    for i, val in enumerate(condition):
        if val:
            count += 1
            if count >= min_samples:
                return i - count + 1   # start of the run
        else:
            count = 0
    return None


# =============================================================================
# LOAD FILE
# =============================================================================

def load_file(path: str) -> Tuple[List[str], pd.DataFrame, float]:
    """
    Returns
    -------
    comments : list of str  — the % header lines
    df       : DataFrame    — data with Timestamp, Channel1-4, Marker
    fs       : float        — sample rate parsed from comments (default 250)
    """
    comments = []
    fs = 250.0
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                comments.append(line.rstrip('\n'))
                if 'Sample Rate' in line:
                    try:
                        fs = float(line.split(':')[1].strip().split()[0])
                    except Exception:
                        pass

    df = pd.read_csv(path, comment='%', skip_blank_lines=True)
    df['Marker']    = df['Marker'].fillna('').astype(str).str.strip()
    df['Timestamp'] = df['Timestamp'].astype(float)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    for c in CHANNEL_COLS:
        df[c] = df[c].astype(float)

    return comments, df, fs


# =============================================================================
# GLOBAL NORMALIZATION
# =============================================================================

def global_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize all 4 sensor channels using whole-file stats."""
    df = df.copy()
    for c in CHANNEL_COLS:
        mu  = df[c].mean()
        sig = df[c].std()
        if sig < EPS:
            sig = 1.0
        df[c] = (df[c] - mu) / sig
    return df


# =============================================================================
# REST SAMPLE EXTRACTION
# =============================================================================

def extract_rest_samples(df: pd.DataFrame,
                         fs: float,
                         t_first_press: float,
                         n: int = REST_N,
                         win_ms: float = REST_WIN_MS,
                         skip_sec: float = REST_SKIP_SEC,
                         seed: int = 42) -> List[Tuple[float, float]]:
    """
    Extract n random win_ms snippets from [skip_sec, t_first_press).
    Returns list of (interval_start, interval_end) in seconds.
    """
    win_sec = win_ms / 1000.0
    t_start = df['Timestamp'].iloc[0] + skip_sec
    t_end   = t_first_press - win_sec

    if t_end <= t_start:
        print(f"  WARNING: Rest window too short "
              f"({t_first_press - df['Timestamp'].iloc[0]:.2f}s). "
              f"Need at least {skip_sec + win_sec:.2f}s before first press.")
        return []

    random.seed(seed)
    intervals = []
    duration_ms = (t_end - t_start) * 1000.0

    for _ in range(n):
        offset_ms = random.uniform(0, duration_ms)
        t0 = t_start + offset_ms / 1000.0
        t1 = t0 + win_sec
        intervals.append((round(t0, 6), round(t1, 6)))

    print(f"  Extracted {len(intervals)} rest (A) snippets "
          f"from [{t_start:.3f}s, {t_first_press:.3f}s)")
    return intervals


# =============================================================================
# ACTIVITY SIGNAL
# =============================================================================

def build_activity_signal(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Velocity-only activity signal — sum of absolute first differences across
    all 4 channels. No position term so signal is invariant to hand end position
    and insensitive to slow sensor creep (which produces tiny |dz|).
    seg   : (N, 4) normalized sensor data
    Returns smoothed activity signal A(t), shape (N,)
    """
    dz = np.diff(seg, axis=0, prepend=seg[[0]])   # (N, 4) first differences
    A  = np.sum(np.abs(dz), axis=1)               # sum across channels
    smooth_samples = ms_to_samples(SMOOTH_MS, fs)
    A = moving_average(A, smooth_samples)
    return A


# =============================================================================
# ONSET / OFFSET DETECTION
# =============================================================================

def detect_onset_offset(
    df: pd.DataFrame,
    fs: float,
    t_click: float,
    label: str
) -> Optional[Tuple[float, float]]:
    """
    Detect onset and offset times for a single trial.
    Returns (t_onset, t_offset) in seconds, or None if trial should be rejected.
    """
    # --- baseline window: 300ms before click ---
    t_base_start = t_click - BASELINE_MS / 1000.0
    base_mask    = (df['Timestamp'] >= t_base_start) & (df['Timestamp'] < t_click)
    base_df      = df[base_mask]

    if len(base_df) < 5:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: baseline window too short "
              f"({len(base_df)} samples).")
        return None

    base_arr = base_df[CHANNEL_COLS].to_numpy()

    # T_on from baseline stats — unchanged
    A_base   = build_activity_signal(base_arr, fs)
    mu_A     = A_base.mean()
    sigma_A  = A_base.std() + EPS
    T_on     = mu_A + TON_SIGMA * sigma_A
    # T_off is computed after onset from motion peak — see below

    # --- post-click search window ---
    t_search_start = t_click + SKIP_MS / 1000.0
    t_search_end   = t_click + ONSET_SEARCH_MS / 1000.0
    post_mask      = (df['Timestamp'] >= t_search_start) & \
                     (df['Timestamp'] <= t_search_end + MAX_MOTION_MS / 1000.0)
    post_df        = df[post_mask].reset_index(drop=True)

    if len(post_df) < 5:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: not enough post-click data.")
        return None

    post_arr = post_df[CHANNEL_COLS].to_numpy()
    A_post   = build_activity_signal(post_arr, fs)
    timestamps = post_df['Timestamp'].to_numpy()

    # --- onset: A(t) > T_on for >= ONSET_MIN_MS ---
    onset_min_samples  = ms_to_samples(ONSET_MIN_MS, fs)
    onset_search_end   = np.searchsorted(timestamps, t_search_end)
    onset_condition    = A_post[:onset_search_end] > T_on
    onset_run_start    = continuous_threshold(A_post[:onset_search_end],
                                              onset_condition,
                                              onset_min_samples)

    if onset_run_start is None:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: no onset found within "
              f"{ONSET_SEARCH_MS}ms.")
        return None

    t_onset = timestamps[onset_run_start]

    # --- T_off: peak-relative from 90th percentile of post-onset activity ---
    post_onset_A = A_post[onset_run_start:]
    M_peak       = np.percentile(post_onset_A, PEAK_PERCENTILE)
    T_off        = TOFF_ALPHA * M_peak

    # --- offset: A(t) < T_off for >= OFFSET_MIN_MS, search after onset ---
    offset_min_samples = ms_to_samples(OFFSET_MIN_MS, fs)
    offset_condition   = post_onset_A < T_off
    offset_run_start   = continuous_threshold(post_onset_A,
                                              offset_condition,
                                              offset_min_samples)

    if offset_run_start is None:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: no offset found.")
        return None

    t_offset = timestamps[onset_run_start + offset_run_start]

    # --- sanity check duration ---
    duration_ms = (t_offset - t_onset) * 1000.0
    if duration_ms < MIN_MOTION_MS:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: motion too short "
              f"({duration_ms:.1f}ms < {MIN_MOTION_MS}ms).")
        return None
    if duration_ms > MAX_MOTION_MS:
        print(f"  SKIP [{label} @ {t_click:.3f}s]: motion too long "
              f"({duration_ms:.1f}ms > {MAX_MOTION_MS}ms).")
        return None

    print(f"  OK   [{label} @ {t_click:.3f}s]: onset={t_onset:.3f}s  "
          f"offset={t_offset:.3f}s  duration={duration_ms:.1f}ms")
    return t_onset, t_offset


# =============================================================================
# SLIDING WINDOW
# =============================================================================

def sliding_windows(t_onset: float,
                    t_offset: float,
                    win_ms: float = WINDOW_MS,
                    stride_ms: float = STRIDE_MS) -> List[Tuple[float, float]]:
    """
    Produce (start, end) pairs for 100ms windows with 50ms stride.
    Stops as soon as window_end + stride would exceed t_offset.
    The last accepted window is the one where window_end <= t_offset.
    """
    win_sec    = win_ms    / 1000.0
    stride_sec = stride_ms / 1000.0
    windows    = []
    t0         = t_onset

    while True:
        t1 = t0 + win_sec
        if t1 > t_offset:
            break
        windows.append((round(t0, 6), round(t1, 6)))
        t0 += stride_sec

    return windows


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess(path: str) -> str:
    print("=" * 60)
    print("Air Writing Preprocessing Pipeline")
    print(f"Input: {path}")
    print("=" * 60)

    # 1. Load
    comments, df, fs = load_file(path)
    print(f"\nLoaded {len(df)} samples at {fs} Hz")

    # 2. Global normalize
    df = global_normalize(df)
    print("Global z-score normalization applied.")

    # 3. Find all marker transitions (A/'' -> B/C/D/E)
    motion_labels = {'B', 'C', 'D', 'E'}
    trials = []   # list of (t_click, label)
    prev_marker = ''
    for _, row in df.iterrows():
        m = row['Marker']
        if m in motion_labels and prev_marker not in motion_labels:
            trials.append((row['Timestamp'], m))
        prev_marker = m

    if not trials:
        raise ValueError("No B/C/D/E marker transitions found in file.")

    t_first_press = trials[0][0]
    print(f"\nFound {len(trials)} button press(es). First at t={t_first_press:.3f}s")

    # 4. Rest samples
    print("\n--- Rest samples ---")
    rest_intervals = extract_rest_samples(df, fs, t_first_press)

    # 5. Gesture trials
    print("\n--- Gesture trials ---")
    gesture_intervals = []   # list of (t_start, t_end, label)

    for t_click, label in trials:
        result = detect_onset_offset(df, fs, t_click, label)
        if result is None:
            continue
        t_onset, t_offset = result
        windows = sliding_windows(t_onset, t_offset)
        for (ws, we) in windows:
            gesture_intervals.append((ws, we, label))

    # 6. Assemble all rows: rest first, then gestures
    all_rows = []
    for (t0, t1) in rest_intervals:
        all_rows.append((t0, t1, 'A'))
    all_rows.extend(gesture_intervals)

    # Sort by IntervalStart
    all_rows.sort(key=lambda r: r[0])

    # 7. Write output
    base, _ = os.path.splitext(path)
    out_path = base + '_intervals.txt'

    with open(out_path, 'w') as f:
        # Carry over original comment header
        for c in comments:
            f.write(c + '\n')
        f.write(f'% Preprocessed by m1_preprocess.py\n')
        f.write(f'% Window: {WINDOW_MS}ms, Stride: {STRIDE_MS}ms\n')
        f.write(f'% Onset threshold : mu_baseline + {TON_SIGMA}*sigma_baseline\n')
        f.write(f'% Offset threshold: {TOFF_ALPHA} * M_peak (90th percentile of post-onset activity)\n')
        f.write('IntervalStart,IntervalEnd,Label\n')
        for (t0, t1, lbl) in all_rows:
            f.write(f'{t0:.6f},{t1:.6f},{lbl}\n')

    print(f"\n{'=' * 60}")
    print(f"Output written to: {out_path}")
    print(f"  Rest intervals (A):    {len(rest_intervals)}")
    print(f"  Gesture intervals:     {len(gesture_intervals)}")
    print(f"  Total rows:            {len(all_rows)}")
    print("=" * 60)

    return out_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = input("Enter path to data file: ").strip()
    else:
        path = sys.argv[1]

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    preprocess(path)
