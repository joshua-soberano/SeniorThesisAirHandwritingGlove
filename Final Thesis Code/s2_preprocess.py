"""
Air Writing Letter Interval Preprocessor  (m2_preprocess_intervals.py)
-----------------------------------------------------------------------
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

# =============================================================================
# PARAMETERS  (must match m2_preprocess_letters.py and m2_extract_onsets.py)
# =============================================================================

CHANNEL_COLS   = ['Channel1', 'Channel2', 'Channel3', 'Channel4']
MIN_STROKE_MS   = 400     # reject stroke if duration shorter than this
MAX_STROKE_MS   = 4000    # reject stroke if duration longer than this
DEBOUNCE_MS     = 25      # button must be held this long to register
WINDOW_MS       = 100     # sliding window size
STRIDE_MS       = 50      # sliding window stride

# Onset / offset detection (matching m1_preprocess.py activity signal method)
SMOOTH_MS       = 30      # moving-average window for activity signal (ms)
BASE_PRE_MS     = 300     # baseline window before click (ms)
ONSET_SEARCH_MS = 500     # max ms after click to search for onset
ONSET_STD_MULT  = 4.0     # onset threshold = mu_A + N*sigma_A
ONSET_MIN_MS    = 40      # onset must exceed threshold continuously this long
OFFSET_MIN_MS   = 75      # offset must stay below threshold continuously this long
PEAK_PERCENTILE = 90      # percentile of post-onset activity for offset threshold


# =============================================================================
# HELPERS
# =============================================================================

def ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


# =============================================================================
# FILE LOADING
# =============================================================================

def load_file(path: str):
    """Load raw letter recording. Returns (comments, df, fs, letter)."""
    comments = []
    fs       = 250.0
    letter   = os.path.basename(path).split('_')[0].upper()

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                comments.append(line.rstrip('\n'))
                if 'Sample Rate' in line:
                    try:
                        fs = float(line.split(':')[1].strip().split()[0])
                    except Exception:
                        pass
                if 'Letter' in line:
                    try:
                        letter = line.split(':')[1].strip().upper()
                    except Exception:
                        pass

    df = pd.read_csv(path, comment='%', skip_blank_lines=True)
    df['Timestamp'] = df['Timestamp'].astype(float)
    df['Marker']    = df['Marker'].fillna('').astype(str).str.strip()
    df = df.sort_values('Timestamp').reset_index(drop=True)
    for c in CHANNEL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return comments, df, fs, letter


def normalize_and_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-file z-score normalization then a 3-sample causal moving
    average to each sensor channel. Smoothing is applied after normalization
    to suppress single-sample electrical noise with minimal signal distortion.
    """
    df = df.copy()
    for c in CHANNEL_COLS:
        if c not in df.columns:
            continue
        mu  = df[c].mean()
        sig = df[c].std()
        if sig < 1e-8:
            sig = 1.0
        df[c] = (df[c] - mu) / sig
        df[c] = df[c].rolling(window=3, min_periods=1).mean()
    return df


# =============================================================================
# DEBOUNCED STROKE PAIR DETECTION
# =============================================================================

def find_stroke_pairs(df: pd.DataFrame, fs: float) -> list:
    """
    Find matched B/C -> D/E button pairs with 25ms debounce.
    Returns list of (t_start_press, t_end_press, stroke_idx) tuples.
    """
    start_markers  = {'B', 'C'}
    end_markers    = {'D', 'E'}
    debounce_samp  = ms_to_samples(DEBOUNCE_MS, fs)

    pairs        = []
    in_stroke    = False
    t_start      = None
    stroke_idx   = 0

    cand_marker  = ''
    cand_count   = 0
    cand_start_t = 0.0

    for _, row in df.iterrows():
        t = row['Timestamp']
        m = row['Marker']

        if m in start_markers or m in end_markers:
            if m == cand_marker:
                cand_count += 1
            else:
                cand_marker  = m
                cand_count   = 1
                cand_start_t = t

            if cand_count == debounce_samp:
                if cand_marker in start_markers:
                    if in_stroke:
                        print(f"  WARNING: Start re-pressed at t={cand_start_t:.3f}s "
                              f"before end — resetting.")
                    in_stroke = True
                    t_start   = cand_start_t

                elif cand_marker in end_markers:
                    if not in_stroke:
                        print(f"  WARNING: End pressed at t={cand_start_t:.3f}s "
                              f"with no open stroke — ignoring.")
                    else:
                        stroke_idx += 1
                        pairs.append((t_start, cand_start_t, stroke_idx))
                        in_stroke = False
        else:
            cand_marker = ''
            cand_count  = 0

    if in_stroke:
        print(f"  WARNING: Recording ended with unclosed stroke — discarded.")

    return pairs


# =============================================================================
# ACTIVITY SIGNAL ONSET / OFFSET DETECTION  (matching m1_preprocess.py)
# =============================================================================

def ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def continuous_threshold(signal: np.ndarray,
                         condition: np.ndarray,
                         min_run: int) -> Optional[int]:
    """Return index of start of first run of True in condition >= min_run long."""
    count = 0
    start = None
    for i, c in enumerate(condition):
        if c:
            if start is None:
                start = i
            count += 1
            if count >= min_run:
                return start
        else:
            count = 0
            start = None
    return None


def build_activity_signal(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Sum of absolute first differences across all channels, smoothed
    with a causal moving average. Represents motion velocity.
    """
    diff    = np.abs(np.diff(seg, axis=0))
    activity = diff.sum(axis=1)
    win      = ms_to_samples(SMOOTH_MS, fs)
    kernel   = np.ones(win) / win
    smoothed = np.convolve(activity, kernel, mode='full')[:len(activity)]
    return smoothed


def detect_onset_offset(df: pd.DataFrame, fs: float,
                        t_click: float,
                        stroke_idx: int) -> Optional[Tuple[float, float]]:

    timestamps = df['Timestamp'].to_numpy()
    data       = df[CHANNEL_COLS].to_numpy(dtype=float)

    # baseline: BASE_PRE_MS before click
    t_base_start = t_click - BASE_PRE_MS / 1000.0
    base_mask    = (timestamps >= t_base_start) & (timestamps < t_click)
    base_arr     = data[base_mask]
    if len(base_arr) < 5:
        print(f"  SKIP stroke {stroke_idx}: insufficient baseline samples.")
        return None

    A_base  = build_activity_signal(base_arr, fs)
    mu_A    = A_base.mean()
    sigma_A = A_base.std() + 1e-8
    T_on    = mu_A + ONSET_STD_MULT * sigma_A

    # post-click window to search for onset
    t_search_end  = t_click + ONSET_SEARCH_MS / 1000.0
    post_mask     = (timestamps >= t_click) & (timestamps <= t_search_end)
    post_arr      = data[post_mask]
    post_times    = timestamps[post_mask]
    if len(post_arr) < 5:
        print(f"  SKIP stroke {stroke_idx}: no post-click data.")
        return None

    A_post       = build_activity_signal(post_arr, fs)
    onset_min_s  = ms_to_samples(ONSET_MIN_MS, fs)
    onset_run    = continuous_threshold(A_post, A_post > T_on, onset_min_s)
    if onset_run is None:
        print(f"  SKIP stroke {stroke_idx}: no onset within {ONSET_SEARCH_MS}ms.")
        return None
    t_onset = post_times[onset_run]

    # offset: threshold from peak of post-onset activity
    post_onset_mask = timestamps >= t_onset
    post_onset_arr  = data[post_onset_mask]
    post_onset_times= timestamps[post_onset_mask]
    if len(post_onset_arr) < 5:
        print(f"  SKIP stroke {stroke_idx}: no data after onset.")
        return None

    A_motion  = build_activity_signal(post_onset_arr, fs)
    M_peak    = np.percentile(A_motion, PEAK_PERCENTILE)
    off_mult  = max(0.5, min(2.0, M_peak / (sigma_A + 1e-8) * 0.3))
    T_off     = mu_A + off_mult * sigma_A

    offset_min_s = ms_to_samples(OFFSET_MIN_MS, fs)
    offset_run   = continuous_threshold(A_motion,
                                        A_motion < T_off,
                                        offset_min_s)
    if offset_run is None:
        # fallback: use end of search window
        t_offset = post_onset_times[-1]
    else:
        t_offset = post_onset_times[offset_run]

    duration_ms = (t_offset - t_onset) * 1000.0
    if duration_ms < MIN_STROKE_MS:
        print(f"  SKIP stroke {stroke_idx}: too short ({duration_ms:.1f}ms).")
        return None
    if duration_ms > MAX_STROKE_MS:
        print(f"  SKIP stroke {stroke_idx}: too long ({duration_ms:.1f}ms).")
        return None

    print(f"  OK   stroke {stroke_idx}: onset={t_onset:.3f}s  "
          f"offset={t_offset:.3f}s  duration={duration_ms:.1f}ms")
    return t_onset, t_offset


# =============================================================================
# SLIDING WINDOW
# =============================================================================

def sliding_windows(t_onset: float, t_offset: float) -> List[Tuple[float, float]]:
    """
    100ms windows with 50ms stride over [t_onset, t_offset].
    Stops when window_end > t_offset.
    """
    win_sec    = WINDOW_MS / 1000.0
    stride_sec = STRIDE_MS / 1000.0
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

def preprocess(path: str) -> Optional[str]:
    print("=" * 60)
    print("Air Writing Letter Interval Preprocessor")
    print(f"Input: {path}")
    print("=" * 60)

    comments, df, fs, letter = load_file(path)
    df = normalize_and_smooth(df)
    print(f"\nLetter : {letter}")
    print(f"Samples: {len(df)} at {fs} Hz  (z-score normalized, 3-sample smoothed)")

    # find debounced stroke pairs
    pairs = find_stroke_pairs(df, fs)

    if not pairs:
        print("  No valid stroke pairs found — skipping.")
        return None

    print(f"\nFound {len(pairs)} stroke pair(s).")
    print("\n--- Stroke Processing ---")

    all_intervals = []   # list of (t_start, t_end, letter)
    strokes_ok    = 0
    strokes_skip  = 0

    for (t_start_press, t_end_press, stroke_idx) in pairs:
        result = detect_onset_offset(df, fs, t_start_press, stroke_idx)
        if result is None:
            strokes_skip += 1
            continue

        t_onset, t_offset = result
        windows = sliding_windows(t_onset, t_offset)

        if not windows:
            print(f"  SKIP stroke {stroke_idx}: no valid windows extracted.")
            strokes_skip += 1
            continue

        for (ws, we) in windows:
            all_intervals.append((ws, we, letter))

        print(f"  Stroke {stroke_idx}: {len(windows)} interval(s) extracted.")
        strokes_ok += 1

    if not all_intervals:
        print("\nNo intervals extracted — skipping file.")
        return None

    # sort by start time
    all_intervals.sort(key=lambda r: r[0])

    # write output
    base, _  = os.path.splitext(path)
    out_path = base + '_intervals.txt'

    with open(out_path, 'w') as f:
        for c in comments:
            f.write(c + '\n')
        f.write('% Preprocessed by m2_preprocess_intervals.py\n')
        f.write(f'% Normalization: per-file z-score + 3-sample causal moving average\n')
        f.write(f'% Letter: {letter}\n')
        f.write(f'% Onset:  activity signal > mu + {ONSET_STD_MULT}*sigma for {ONSET_MIN_MS}ms\n')
        f.write(f'% Offset: activity signal < peak-relative threshold for {OFFSET_MIN_MS}ms\n')
        f.write(f'% Duration limits: {MIN_STROKE_MS}ms — {MAX_STROKE_MS}ms\n')
        f.write(f'% Window: {WINDOW_MS}ms, Stride: {STRIDE_MS}ms\n')
        f.write(f'% Strokes OK: {strokes_ok}  Skipped: {strokes_skip}\n')
        f.write('IntervalStart,IntervalEnd,Label\n')
        for (t0, t1, lbl) in all_intervals:
            f.write(f'{t0:.6f},{t1:.6f},{lbl}\n')

    print(f"\n{'=' * 60}")
    print(f"Output written to: {out_path}")
    print(f"  Strokes OK      : {strokes_ok}")
    print(f"  Strokes skipped : {strokes_skip}")
    print(f"  Total intervals : {len(all_intervals)}")
    print("=" * 60)

    return out_path


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) < 2:
        path = input("Enter path to letter recording file or directory: ").strip()
    else:
        path = sys.argv[1]

    if os.path.isdir(path):
        # batch process all .txt files in directory
        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.txt')
            and not f.endswith('_intervals.txt')
            and not f.endswith('_onsets.txt')
            and not f.endswith('_sequences.txt')
        ])
        if not files:
            print("No valid .txt recording files found in directory.")
            sys.exit(1)
        print(f"Found {len(files)} recording file(s).\n")
        success = 0
        skipped = 0
        for f in files:
            result = preprocess(f)
            if result:
                success += 1
            else:
                skipped += 1
        print(f"\nDone. {success} file(s) processed, {skipped} skipped.")

    elif os.path.isfile(path):
        preprocess(path)

    else:
        print(f"Path not found: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
