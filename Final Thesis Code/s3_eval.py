"""
m5_eval.py — Evaluate a saved Word GRU checkpoint with the decoder
-------------------------------------------------------------------
Loads a trained checkpoint produced by m5_train.py and runs
greedy or beam search decoding on the saved test set.

Expects these files saved by m5_train.py (all with same timestamp):
    m5_gru_{ts}.pt
    m5_gru_{ts}_classes.npy
    m5_gru_{ts}_wordlist.npy
    m5_gru_{ts}_avg_intervals.npy
    m5_gru_{ts}_testset.npy

"""

import os
import sys
import numpy as np
from datetime import datetime
import torch

from m5_common import (
    WordGRU, WordGRU_CTC,
    decode_and_evaluate, decode_and_evaluate_ctc,
    save_results,
)
from m5_decoder import BeamDecoder, LanguageModel


def load_checkpoint(data_dir: str, ts: str):
    """Load model, letter classes, word list, avg_intervals and test set."""
    pt_path  = os.path.join(data_dir, f'm5_gru_{ts}.pt')
    cls_path = os.path.join(data_dir, f'm5_gru_{ts}_classes.npy')
    wl_path  = os.path.join(data_dir, f'm5_gru_{ts}_wordlist.npy')
    ai_path  = os.path.join(data_dir, f'm5_gru_{ts}_avg_intervals.npy')
    ts_path  = os.path.join(data_dir, f'm5_gru_{ts}_testset.npy')
    type_path = os.path.join(data_dir, f'm5_gru_{ts}_type.npy')

    for p in [pt_path, cls_path, wl_path, ai_path, ts_path]:
        if not os.path.isfile(p):
            print(f"Missing file: {p}")
            sys.exit(1)

    letter_classes       = list(np.load(cls_path, allow_pickle=True))
    valid_words          = list(np.load(wl_path,  allow_pickle=True))
    avg_letter_intervals = float(np.load(ai_path)[0])
    test_s               = list(np.load(ts_path,  allow_pickle=True))

    # load model type — default to 'boundary' for old checkpoints without type file
    if os.path.isfile(type_path):
        model_type = str(np.load(type_path, allow_pickle=True)[0])
    else:
        model_type = 'boundary'

    n_letters = len(letter_classes)
    if model_type == 'ctc':
        model = WordGRU_CTC(n_letters)
    else:
        model = WordGRU(n_letters)

    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model.eval()

    return model, letter_classes, valid_words, avg_letter_intervals, test_s, model_type


def main():
    print("=" * 60)
    print("Air Writing Word Model — Evaluation  (m5_eval.py)")
    print("=" * 60)

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = input("\nEnter path to sequences directory (where checkpoint is saved): ").strip()

    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    # --- find available checkpoints ---
    pts = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith('m5_gru_') and f.endswith('.pt')
    ])
    if not pts:
        print(f"No m5_gru_*.pt checkpoints found in {data_dir}")
        sys.exit(1)

    print("\nAvailable checkpoints:")
    for i, p in enumerate(pts, 1):
        print(f"  {i}. {p}")

    if len(pts) == 1:
        chosen = pts[0]
        print(f"  Using: {chosen}")
    else:
        idx = int(input("  Select checkpoint number: ").strip()) - 1
        chosen = pts[idx]

    # extract timestamp from filename e.g. m5_gru_20260407_1537.pt
    ts = chosen.replace('m5_gru_', '').replace('.pt', '')

    print(f"\nLoading checkpoint: {chosen}")
    model, letter_classes, valid_words, avg_letter_intervals, test_s, model_type = \
        load_checkpoint(data_dir, ts)

    print(f"  Model type    : {model_type}")
    print(f"  Letter classes ({len(letter_classes)}): {letter_classes}")
    print(f"  Word list size: {len(valid_words)}")
    print(f"  Test samples  : {len(test_s)}")
    print(f"  Avg letter duration: {avg_letter_intervals:.1f} intervals "
          f"({avg_letter_intervals * 50:.0f}ms)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model.to(device)

    # --- CTC model: simpler eval path ---
    if model_type == 'ctc':
        print("\n[CTC decoder configuration]")
        print("  1. Greedy (fast)")
        print("  2. Beam search with language model")
        ctc_dec = input("  Choose decoder (1/2): ").strip()
        use_ctc_beam = ctc_dec == '2'

        lm        = None
        lm_path   = 'N/A'
        lm_weight = 0.3
        bw        = 10

        if use_ctc_beam:
            lm_path   = input("  Path to LM file (.pkl from m5_build_lm.py, blank = char n-gram from word list): ").strip()
            bw        = int(input("  Beam width (default 10): ").strip() or "10")
            lm_weight = float(input("  LM weight for beam scoring (default 0.0, LM still used in word correction): ").strip() or "0.0")
            if lm_path and os.path.isfile(lm_path):
                if lm_path.endswith('.pkl'):
                    lm = LanguageModel.from_pkl(lm_path, letter_classes)
                else:
                    lm = LanguageModel.from_binary(lm_path, letter_classes)
            else:
                lm = LanguageModel.uniform(letter_classes)

        length_tolerance = int(input("  Word length tolerance (default 1 for CTC — decoded length is reliable): ").strip() or "1")

        results, word_acc, mean_cer, letter_acc = decode_and_evaluate_ctc(
            model, test_s, letter_classes, valid_words, device,
            avg_letter_intervals = avg_letter_intervals,
            length_tolerance     = length_tolerance,
            use_beam             = use_ctc_beam,
            beam_width           = bw,
            lm                   = lm,
            lm_weight            = lm_weight,
        )

        cor_acc = sum(r['correct_corrected'] for r in results) / len(results)
        raw_acc = sum(r['correct_raw']       for r in results) / len(results)
        cor_cer = sum(r['cer_corrected']     for r in results) / len(results)
        raw_cer = sum(r['cer_raw']           for r in results) / len(results)

        print(f"\n  {'':30}  {'Word Acc':>10}  {'Mean CER':>10}")
        print(f"  {'CTC raw':30}  {raw_acc:>10.4f}  {raw_cer:>10.4f}")
        print(f"  {'After word correction':30}  {cor_acc:>10.4f}  {cor_cer:>10.4f}")

        eval_ts    = datetime.now().strftime("%Y%m%d_%H%M")
        decoder_name = f'ctc_beam_w{bw}' if use_ctc_beam else 'ctc_greedy'
        run_config = {
            'data_dir'            : data_dir,
            'checkpoint_ts'       : ts,
            'decoder'             : decoder_name,
            'length_tolerance'    : length_tolerance,
            'avg_letter_intervals': avg_letter_intervals,
            'beam_width'          : bw,
            'lm_path'             : lm_path,
            'lm_weight'           : lm_weight,
            'lr': 'N/A', 'n_epochs': 'N/A', 'lam': 'N/A',
            'w_start': 0, 'w_inside': 0, 'w_end': 0, 'w_trans': 0,
            'end_threshold': 'N/A', 'lookahead': 'N/A',
            'alpha': 'N/A', 'beta': 'N/A', 'gamma': 'N/A',
        }
        save_results(results, word_acc, mean_cer, letter_acc,
                     letter_classes, data_dir, eval_ts, run_config)

        print("\n" + "=" * 60)
        print("Done.")
        print("=" * 60)
        return

    # --- boundary model: full decoder config ---
    print("\n[Decoder configuration]")
    print("  1. Greedy (fast)")
    print("  2. Beam search")
    dec_choice = input("  Choose decoder (1/2): ").strip()
    use_beam   = dec_choice == '2'

    lm      = LanguageModel.uniform(letter_classes)
    alpha   = beta = gamma = 1.0
    lm_path = 'N/A'

    if use_beam:
        lm_path  = input("  Path to KenLM binary (blank for uniform LM): ").strip()
        alpha    = float(input("  Alpha (default 1.0): ").strip() or "1.0")
        beta     = float(input("  Beta  (default 1.0): ").strip() or "1.0")
        gamma    = float(input("  Gamma (default 0.0): ").strip() or "0.0")
        bw       = int(input("  Beam width (default 2): ").strip() or "2")
        la       = int(input("  Lookahead (default 0): ").strip() or "0")
        max_beam = int(input("  Max beam samples (default -1 for all): ").strip() or "-1")
        end_threshold    = float(input("  END threshold (default 0.25): ").strip() or "0.25")
        length_tolerance = int(input("  Word length tolerance (default 2): ").strip() or "2")
        if lm_path and os.path.isfile(lm_path):
            lm = LanguageModel.from_binary(lm_path, letter_classes)
    else:
        bw  = 2;  la = 0;  max_beam = 0
        end_threshold    = float(input("  END threshold (default 0.25): ").strip() or "0.25")
        length_tolerance = int(input("  Word length tolerance (default 2): ").strip() or "2")

    # --- evaluate ---
    print("\n" + "=" * 60)
    print(f"Evaluation on test set via {'beam search' if use_beam else 'greedy'} decoder")
    print("=" * 60)

    results, word_acc, mean_cer, letter_acc = decode_and_evaluate(
        model, test_s, letter_classes, lm, device,
        word_list            = valid_words,
        avg_letter_intervals = avg_letter_intervals,
        use_beam             = use_beam,
        beam_width           = bw, lookahead = la,
        alpha=alpha, beta=beta, gamma=gamma,
        end_threshold        = end_threshold,
        length_tolerance     = length_tolerance,
        max_beam_samples     = max_beam,
    )

    raw_acc = sum(r['correct_raw']       for r in results) / len(results)
    col_acc = sum(r['correct_collapse']  for r in results) / len(results)
    cor_acc = sum(r['correct_corrected'] for r in results) / len(results)
    raw_cer = sum(r['cer_raw']           for r in results) / len(results)
    col_cer = sum(r['cer_collapse']      for r in results) / len(results)
    cor_cer = sum(r['cer_corrected']     for r in results) / len(results)

    print(f"\n  Letter accuracy (at true END positions) : {letter_acc:.4f}")
    print(f"\n  {'':30}  {'Word Acc':>10}  {'Mean CER':>10}")
    print(f"  {'Raw greedy':30}  {raw_acc:>10.4f}  {raw_cer:>10.4f}")
    print(f"  {'After collapse':30}  {col_acc:>10.4f}  {col_cer:>10.4f}")
    print(f"  {'After word correction':30}  {cor_acc:>10.4f}  {cor_cer:>10.4f}")

    print("\n  Sample predictions (first 20):")
    print(f"  {'True':>12}  {'Raw':>12}  {'Collapsed':>12}  {'Corrected':>12}  {'OK':>5}  {'CER':>6}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*5}  {'-'*6}")
    for r in results[:20]:
        print(f"  {r['true_word']:>12}  {r['decoded_raw']:>12}  "
              f"{r['decoded_collapse']:>12}  {r['decoded_corrected']:>12}  "
              f"{'YES' if r['correct_corrected'] else 'no':>5}  "
              f"{r['cer_corrected']:>6.3f}")

    # --- save results ---
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_config = {
        'data_dir'            : data_dir,
        'checkpoint_ts'       : ts,
        'decoder'             : 'beam' if use_beam else 'greedy',
        'end_threshold'       : end_threshold,
        'length_tolerance'    : length_tolerance,
        'avg_letter_intervals': avg_letter_intervals,
        'beam_width'          : bw,
        'lookahead'           : la,
        'alpha'               : alpha,
        'beta'                : beta,
        'gamma'               : gamma,
        'lm_path'             : lm_path,
        # placeholders for save_results compatibility
        'lr'    : 'N/A (loaded checkpoint)',
        'n_epochs': 'N/A (loaded checkpoint)',
        'lam'   : 'N/A (loaded checkpoint)',
        'w_start': 3.0, 'w_inside': 0.5, 'w_end': 5.0, 'w_trans': 2.0,
    }

    save_results(results, word_acc, mean_cer, letter_acc,
                 letter_classes, data_dir, eval_ts, run_config)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
