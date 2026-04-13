"""
m5_train.py — Train the Word GRU and save checkpoint
------------------------------------------------------
Loads _sequences.txt files, builds synthetic word sequences,
trains the GRU, and saves:
    m5_gru_{timestamp}.pt
    m5_gru_{timestamp}_classes.npy
    m5_gru_{timestamp}_wordlist.npy   (valid words for eval)
    m5_gru_{timestamp}_avg_intervals.npy  (avg letter duration)
    m5_gru_{timestamp}_testset.npy    (test samples for eval)

Usage
-----
python m5_train.py /path/to/sequences/dir
python m5_train.py                          # prompts
"""

import os
import sys
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

from m5_common import (
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    load_all_strokes, build_word_sequences, make_loader,
    WordGRU, WordGRU_CTC, train_model, train_model_ctc,
    save_checkpoint, COMMON_WORDS,
)


def main():
    print("=" * 60)
    print("Air Writing Word Model — Training  (m5_train.py)")
    print("=" * 60)

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = input("\nEnter path to sequences directory: ").strip()

    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    # --- load strokes ---
    print(f"\nLoading sequences from: {data_dir}")
    strokes_by_letter = load_all_strokes(data_dir)
    available_letters = set(strokes_by_letter.keys())

    # --- filter word list ---
    valid_words = [
        w.upper() for w in COMMON_WORDS
        if all(c.upper() in available_letters for c in w)
        and len(w) >= 2
    ]
    print(f"Words usable with available letters: {len(valid_words)} / {len(COMMON_WORDS)}")
    if not valid_words:
        print("ERROR: No words can be built from available letters.")
        sys.exit(1)

    # --- encode labels ---
    all_letters    = sorted(available_letters)
    le             = LabelEncoder()
    le.fit(all_letters)
    letter_classes = list(le.classes_)
    n_letters      = len(letter_classes)
    print(f"Letter classes ({n_letters}): {letter_classes}")

    # --- build synthetic word sequences ---
    print("\nBuilding synthetic word sequences...")
    num_repeats = int(input(
        "  Repeats per word (default 3, each uses different stroke combos): "
    ).strip() or "3")
    samples = build_word_sequences(strokes_by_letter, valid_words,
                                   num_repeats=num_repeats)

    for s in samples:
        s['letters_enc'] = list(le.transform(s['letters']))

    if not samples:
        print("ERROR: No samples built.")
        sys.exit(1)

    # --- 70/15/15 split ---
    idx = np.arange(len(samples))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED)

    train_s = [samples[i] for i in idx_train]
    val_s   = [samples[i] for i in idx_val]
    test_s  = [samples[i] for i in idx_test]

    print(f"\nSplit — Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    train_loader = make_loader(train_s, shuffle=True)
    val_loader   = make_loader(val_s,   shuffle=False)

    # --- compute avg letter stroke duration ---
    stroke_intervals = []
    for s in train_s:
        offsets = s['offsets']
        for i, off in enumerate(offsets):
            if i == 0:
                stroke_len = off + 1
            else:
                gap = off - offsets[i-1]
                stroke_len = max(1, gap - 3)
            stroke_intervals.append(stroke_len)

    avg_letter_intervals = float(np.median(stroke_intervals)) if stroke_intervals else 20.0
    print(f"\n  Average letter stroke duration (median, excl. transitions): "
          f"{avg_letter_intervals:.1f} timesteps ({avg_letter_intervals * 50:.0f}ms)")

    # --- hyperparameters ---
    print("\n[Loss mode]")
    print("  1. Boundary-aware (CrossEntropy letter + boundary heads)")
    print("  2. CTC (no boundary labels — segmentation implicit)")
    loss_choice = input("  Choose loss mode (1/2): ").strip()
    use_ctc     = loss_choice == '2'

    print(f"\n[GRU configuration]")
    n_epochs = int(input("  Epochs       : ").strip())
    lr       = float(input("  Learning rate: ").strip())
    if not use_ctc:
        lam = float(input("  Lambda (boundary loss weight): ").strip())
    else:
        lam = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # --- train ---
    if use_ctc:
        model      = WordGRU_CTC(n_letters)
        model      = train_model_ctc(model, train_loader, val_loader,
                                     n_epochs, lr, device, letter_classes)
        model_type = 'ctc'
    else:
        model      = WordGRU(n_letters)
        model      = train_model(model, train_loader, val_loader,
                                 n_epochs, lr, lam, device)
        model_type = 'boundary'

    # --- save checkpoint + metadata ---
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    save_checkpoint(model, letter_classes, data_dir, ts)

    # save model type so eval knows which decoder to use
    np.save(os.path.join(data_dir, f'm5_gru_{ts}_type.npy'),
            np.array([model_type]))
    np.save(os.path.join(data_dir, f'm5_gru_{ts}_wordlist.npy'),
            np.array(valid_words))
    np.save(os.path.join(data_dir, f'm5_gru_{ts}_avg_intervals.npy'),
            np.array([avg_letter_intervals]))
    np.save(os.path.join(data_dir, f'm5_gru_{ts}_testset.npy'),
            np.array(test_s, dtype=object), allow_pickle=True)

    print(f"\n  Model type   : {model_type}")
    print(f"  Word list    : m5_gru_{ts}_wordlist.npy")
    print(f"  Avg intervals: m5_gru_{ts}_avg_intervals.npy")
    print(f"  Test set     : m5_gru_{ts}_testset.npy")

    print("\n" + "=" * 60)
    print(f"Training complete. Use m5_eval.py with timestamp {ts} to evaluate.")
    print("=" * 60)


if __name__ == '__main__':
    main()
