"""
Air Writing Letter Model  (m2_model_oneletter.py)
----------------------------------------
Trains a sequence model to classify air-written letters from
direction likelihood sequences produced by m2_preprocess_letters.py.

Input  : directory of {letter}_{timestamp}_sequences.txt files
Output : m2_results_{modelname}_{timestamp}.csv

Models
------
1. GRU
2. LSTM
3. Transformer

Each model predicts letter (A-Z) as primary task.
Boundary detection (offset flag) is an auxiliary loss task.

Combined loss
-------------
total_loss = letter_loss + lambda * boundary_loss

where lambda is configurable at runtime.
"""

import os
import sys
import math
import copy
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED         = 42
TEST_RATIO          = 0.15    # held-out test set
N_FOLDS             = 3       # k-fold CV on remaining 85%
EARLY_STOP_PATIENCE = 10
INPUT_DIM           = 9      # 9 direction probability columns
HIDDEN_SIZE         = 128
N_LAYERS            = 2
DROPOUT             = 0.3
D_MODEL             = 64     # transformer model dim
N_HEADS             = 4
FF_DIM              = 128
TRANSFORMER_DROPOUT = 0.1

DIR_CLASSES = ['E', 'N', 'NE', 'NW', 'REST', 'S', 'SE', 'SW', 'W']

MODEL_MENU = {
    '1': 'GRU',
    '2': 'LSTM',
    '3': 'Transformer',
}
MODEL_SLUGS = {
    '1': 'gru',
    '2': 'lstm',
    '3': 'transformer',
}


# =============================================================================
# DATA LOADING & PARSING
# =============================================================================

def parse_sequences_file(path: str) -> list:
    """
    Parse a _sequences.txt file into a list of stroke dicts:
        {
          'X'        : np.ndarray (K, 9)   direction probs
          'boundary' : np.ndarray (K,)     1.0=offset, 0.0=other
          'letter'   : str                 e.g. 'A'
        }
    """
    strokes  = []
    in_stroke = False
    rows      = []
    letter    = None

    # Extract letter from filename as fallback
    fname_letter = os.path.basename(path).split('_')[0].upper()

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # comment lines
            if line.startswith('%'):
                if 'Stroke:' in line:
                    # save previous stroke if any
                    if in_stroke and rows:
                        stroke = _build_stroke(rows, letter or fname_letter)
                        if stroke is not None:
                            strokes.append(stroke)
                    rows      = []
                    in_stroke = True
                elif 'Letter:' in line:
                    try:
                        letter = line.split(':')[1].strip().upper()
                    except Exception:
                        pass
                continue

            # column header line
            if line.startswith('IntervalStart'):
                continue

            # data row
            if in_stroke and line.strip():
                rows.append(line.strip())

    # save last stroke
    if in_stroke and rows:
        stroke = _build_stroke(rows, letter or fname_letter)
        if stroke is not None:
            strokes.append(stroke)

    return strokes


def _build_stroke(rows: list, letter: str) -> dict:
    """Parse raw row strings into arrays. Returns None if invalid."""
    X_list  = []
    b_list  = []

    BOUNDARY_MAP = {'start': 0, 'inside': 1, 'end': 2, 'transition': 3}

    for row in rows:
        parts = row.split(',')
        # columns: IntervalStart, IntervalEnd, p_E..p_W (9), letter, boundary
        if len(parts) < 12:
            continue
        try:
            probs    = [float(parts[i]) for i in range(2, 11)]   # 9 probs
            ltr      = parts[11].strip().upper()
            boundary_str = parts[12].strip().lower() if len(parts) > 12 else 'inside'
            boundary = BOUNDARY_MAP.get(boundary_str, 1)  # default inside=1
        except (ValueError, IndexError):
            continue

        # use per-row letter if available, else fallback
        if ltr and ltr.isalpha() and len(ltr) == 1:
            letter = ltr

        X_list.append(probs)
        b_list.append(boundary)

    if len(X_list) < 1:
        return None

    return {
        'X'       : np.array(X_list, dtype=np.float32),    # (K, 9)
        'boundary': np.array(b_list, dtype=np.int64),       # (K,) — 0=start, 1=inside, 2=end
        'letter'  : letter,
    }


def load_all_sequences(directory: str) -> list:
    """Load all _sequences.txt files from directory. Returns list of stroke dicts."""
    files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('_sequences.txt')
    ])

    if not files:
        print(f"No _sequences.txt files found in: {directory}")
        sys.exit(1)

    all_strokes = []
    for fpath in files:
        strokes = parse_sequences_file(fpath)
        print(f"  {os.path.basename(fpath)}: {len(strokes)} stroke(s)")
        all_strokes.extend(strokes)

    print(f"\nTotal strokes loaded: {len(all_strokes)}")
    return all_strokes


# =============================================================================
# DATASET
# =============================================================================

class LetterDataset(Dataset):
    def __init__(self, strokes: list, label_encoder: LabelEncoder):
        self.strokes       = strokes
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        s      = self.strokes[idx]
        X      = torch.tensor(s['X'],        dtype=torch.float32)   # (K, 9)
        b      = torch.tensor(s['boundary'], dtype=torch.long)      # (K,) int64
        y      = int(self.label_encoder.transform([s['letter']])[0])
        length = X.shape[0]
        return X, b, y, length


def collate_fn(batch):
    """Pad variable-length sequences to max length in batch."""
    Xs, bs, ys, lengths = zip(*batch)
    max_len = max(lengths)
    B       = len(Xs)

    X_pad = torch.zeros(B, max_len, INPUT_DIM)
    b_pad = torch.full((B, max_len), fill_value=-1, dtype=torch.long)  # -1 = padding (ignored)

    for i, (x, b_) in enumerate(zip(Xs, bs)):
        k = x.shape[0]
        X_pad[i, :k, :] = x
        b_pad[i, :k]    = b_

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    ys_t      = torch.tensor(ys,      dtype=torch.long)
    return X_pad, b_pad, ys_t, lengths_t


# =============================================================================
# POSITIONAL ENCODING (Transformer)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe           = torch.zeros(max_len, d_model)
        pos          = torch.arange(0, max_len).unsqueeze(1).float()
        div          = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class GRUModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = N_LAYERS,
            dropout     = DROPOUT,
            batch_first = True,
        )
        self.letter_head   = nn.Linear(HIDDEN_SIZE, n_classes)
        self.boundary_head = nn.Linear(HIDDEN_SIZE, 4)  # START/INSIDE/END/TRANSITION

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)   # (B, T, H)

        # gather hidden state at last real timestep
        idx          = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
        idx          = idx.expand(-1, 1, HIDDEN_SIZE).to(out.device)
        last_hidden  = out.gather(1, idx).squeeze(1)          # (B, H)

        letter_logits   = self.letter_head(last_hidden)        # (B, n_classes)
        boundary_logits = self.boundary_head(out)               # (B, T, 4)
        return letter_logits, boundary_logits


class LSTMModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_SIZE,
            num_layers  = N_LAYERS,
            dropout     = DROPOUT,
            batch_first = True,
        )
        self.letter_head   = nn.Linear(HIDDEN_SIZE, n_classes)
        self.boundary_head = nn.Linear(HIDDEN_SIZE, 4)  # START/INSIDE/END/TRANSITION

    def forward(self, x, lengths):
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                           enforce_sorted=False)
        out, _      = self.lstm(packed)
        out, _      = pad_packed_sequence(out, batch_first=True)   # (B, T, H)

        idx         = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
        idx         = idx.expand(-1, 1, HIDDEN_SIZE).to(out.device)
        last_hidden = out.gather(1, idx).squeeze(1)                # (B, H)

        letter_logits   = self.letter_head(last_hidden)
        boundary_logits = self.boundary_head(out)               # (B, T, 4)
        return letter_logits, boundary_logits


class TransformerModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, dropout=TRANSFORMER_DROPOUT)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,
            nhead           = N_HEADS,
            dim_feedforward = FF_DIM,
            dropout         = TRANSFORMER_DROPOUT,
            batch_first     = True,
        )
        self.encoder       = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.letter_head   = nn.Linear(D_MODEL, n_classes)
        self.boundary_head = nn.Linear(D_MODEL, 4)  # START/INSIDE/END/TRANSITION

    def forward(self, x, lengths):
        B, T, _ = x.shape

        # padding mask: True where position is padding (should be ignored)
        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
                   lengths.unsqueeze(1).to(x.device)              # (B, T)

        x   = self.input_proj(x)                                  # (B, T, D)
        x   = self.pos_enc(x)
        out = self.encoder(x, src_key_padding_mask=pad_mask)      # (B, T, D)

        # masked mean pooling over real timesteps
        mask_float  = (~pad_mask).unsqueeze(-1).float()           # (B, T, 1)
        pooled      = (out * mask_float).sum(dim=1) / \
                      mask_float.sum(dim=1).clamp(min=1)          # (B, D)

        letter_logits   = self.letter_head(pooled)                # (B, n_classes)
        boundary_logits = self.boundary_head(out)                  # (B, T, 4)
        return letter_logits, boundary_logits


# =============================================================================
# TRAINING
# =============================================================================

def make_loader(dataset: LetterDataset, batch_size: int = 32,
                shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                n_epochs: int,
                lr: float,
                lam: float,
                device: torch.device,
                model_name: str) -> nn.Module:
    """
    Train with combined loss:
        total = CrossEntropy(letter) + lam * BCE(boundary)
    Prints per-epoch: train loss, val loss, val accuracy.
    Early stopping on val loss with patience=10.
    Returns best model.
    """
    optimizer      = torch.optim.Adam(model.parameters(), lr=lr)
    letter_crit    = nn.CrossEntropyLoss()
    # END upweighted — most important for decoder, underrepresented in each stroke
    boundary_weights = torch.tensor([3.0, 0.5, 5.0, 2.0]).to(device)  # START, INSIDE, END, TRANSITION
    boundary_crit  = nn.CrossEntropyLoss(ignore_index=-1, weight=boundary_weights)
    best_val_loss  = float('inf')
    best_weights   = copy.deepcopy(model.state_dict())
    patience_ctr   = 0
    history        = {'train_loss': [], 'val_loss': [], 'val_acc': [],
                      'best_val_loss': float('inf'), 'best_epoch': 0}

    model.to(device)

    print(f"\n  Training {model_name} | epochs={n_epochs} lr={lr} lambda={lam}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*8}")

    for epoch in range(1, n_epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_pad, b_pad, ys, lengths in train_loader:
            X_pad, b_pad = X_pad.to(device), b_pad.to(device)
            ys, lengths  = ys.to(device),    lengths.to(device)

            optimizer.zero_grad()
            letter_logits, boundary_logits = model(X_pad, lengths)

            l_loss = letter_crit(letter_logits, ys)

            # boundary loss: CrossEntropy over (B*T, 4), padding=-1 ignored
            # boundary_logits: (B, T, 4) -> (B*T, 4)
            # b_pad:           (B, T)    -> (B*T,)   (-1 for padding)
            b_loss = boundary_crit(
                boundary_logits.reshape(-1, 4),
                b_pad.reshape(-1)
            )

            loss = l_loss + lam * b_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ys)

        train_loss /= len(train_loader.dataset)

        # --- val ---
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        with torch.no_grad():
            for X_pad, b_pad, ys, lengths in val_loader:
                X_pad, b_pad = X_pad.to(device), b_pad.to(device)
                ys, lengths  = ys.to(device),    lengths.to(device)

                letter_logits, boundary_logits = model(X_pad, lengths)
                l_loss = letter_crit(letter_logits, ys)

                b_loss = boundary_crit(
                    boundary_logits.reshape(-1, 4),
                    b_pad.reshape(-1)
                )

                val_loss    += (l_loss + lam * b_loss).item() * len(ys)
                val_correct += (letter_logits.argmax(1) == ys).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc   = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_acc:>8.4f}")

        if val_loss < best_val_loss:
            best_val_loss            = val_loss
            history['best_val_loss'] = val_loss
            history['best_epoch']    = epoch
            best_weights             = copy.deepcopy(model.state_dict())
            patience_ctr             = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


def train_model_fixed_epochs(model: nn.Module,
                              train_loader: DataLoader,
                              n_epochs: int,
                              lr: float,
                              lam: float,
                              device: torch.device,
                              model_name: str) -> tuple:
    """
    Train for exactly n_epochs with no early stopping — no val set needed.
    Used for the final model where epoch count is determined from CV.
    Returns (model, history).
    """
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
    letter_crit   = nn.CrossEntropyLoss()
    boundary_weights = torch.tensor([3.0, 0.5, 5.0, 2.0]).to(device)
    boundary_crit = nn.CrossEntropyLoss(ignore_index=-1, weight=boundary_weights)
    history       = {'train_loss': [], 'val_loss': [], 'val_acc': [],
                     'best_val_loss': float('nan'), 'best_epoch': n_epochs}

    model.to(device)

    print(f"\n  Training {model_name} (final) | epochs={n_epochs} lr={lr} lambda={lam}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}")
    print(f"  {'-'*6}  {'-'*12}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_pad, b_pad, ys, lengths in train_loader:
            X_pad, b_pad = X_pad.to(device), b_pad.to(device)
            ys, lengths  = ys.to(device),    lengths.to(device)
            optimizer.zero_grad()
            letter_logits, boundary_logits = model(X_pad, lengths)
            l_loss = letter_crit(letter_logits, ys)
            b_loss = boundary_crit(
                boundary_logits.reshape(-1, 4),
                b_pad.reshape(-1)
            )
            loss = l_loss + lam * b_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ys)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        print(f"  {epoch:>6}  {train_loss:>12.4f}")

    return model, history


# =============================================================================
# EVALUATION & SAVING
# =============================================================================

def predict(model: nn.Module, loader: DataLoader,
            device: torch.device) -> tuple:
    """Returns (true labels, predicted labels, probability arrays)."""
    model.eval()
    all_true  = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_pad, b_pad, ys, lengths in loader:
            X_pad   = X_pad.to(device)
            lengths = lengths.to(device)
            logits, _ = model(X_pad, lengths)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()
            preds     = probs.argmax(axis=1)
            all_true.extend(ys.numpy())
            all_preds.extend(preds)
            all_probs.append(probs)

    return (np.array(all_true),
            np.array(all_preds),
            np.concatenate(all_probs, axis=0))


def save_model(model: nn.Module, model_slug: str,
               letter_classes: list, out_dir: str) -> str:
    """Save model state dict as .pt file. Returns saved path."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(out_dir, f'm2_{model_slug}_{ts}.pt')
    torch.save({
        'state_dict'    : model.state_dict(),
        'letter_classes': letter_classes,
        'input_dim'     : INPUT_DIM,
    }, path)
    print(f"  Model saved: {path}")
    return path


def build_eval_lines(y_true, y_pred, letter_classes,
                     model_name: str, config_str: str,
                     history: dict, fold_stats: list,
                     n_trainval: int, n_test: int) -> list:
    """Build comprehensive evaluation lines for printing and saving."""
    sep   = "=" * 70
    lines = []

    lines.append(sep)
    lines.append(f"EVALUATION — {model_name}")
    lines.append(f"Config     : {config_str}")
    lines.append(sep)

    # ── Dataset split ─────────────────────────────────────────────────────────
    lines.append(f"\nDataset split  : train+val={n_trainval} ({N_FOLDS}-fold CV)  test={n_test}")
    lines.append(f"Letters        : {letter_classes}  ({len(letter_classes)} classes)")

    # ── Test accuracy ─────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    n   = len(y_true)
    lines.append(f"\n{'─'*70}")
    lines.append("TEST SET RESULTS")
    lines.append(f"{'─'*70}")
    lines.append(f"Overall accuracy : {acc:.4f}  ({int(acc*n)}/{n} correct)")

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(letter_classes))), zero_division=0
    )
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    wt_p,  wt_r,  wt_f1,  _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    lines.append(f"Macro   P/R/F1   : {mac_p:.4f} / {mac_r:.4f} / {mac_f1:.4f}")
    lines.append(f"Weighted P/R/F1  : {wt_p:.4f} / {wt_r:.4f} / {wt_f1:.4f}")

    # ── Per-letter breakdown ──────────────────────────────────────────────────
    lines.append(f"\n{'─'*70}")
    lines.append("PER-LETTER BREAKDOWN")
    lines.append(f"{'─'*70}")
    hdr = f"  {'Letter':6s}  {'Acc':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>7}  {'Support':>8}"
    lines.append(hdr)
    lines.append("  " + "-" * 56)
    for i, cls in enumerate(letter_classes):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        cls_acc = accuracy_score(y_true[mask], y_pred[mask])
        lines.append(
            f"  {cls:6s}  {cls_acc:>7.4f}  {prec[i]:>10.4f}  {rec[i]:>8.4f}  "
            f"{f1[i]:>7.4f}  {int(sup[i]):>8}"
        )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    lines.append(f"\n{'─'*70}")
    lines.append("CONFUSION MATRIX  (rows=true, cols=pred)")
    lines.append(f"{'─'*70}")
    cm     = confusion_matrix(y_true, y_pred)
    header = "     " + " ".join(f"{c:>3}" for c in letter_classes)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"  {letter_classes[i]:>2}   " + " ".join(f"{v:>3}" for v in row))

    # ── Cross-validation results ─────────────────────────────────────────────
    if fold_stats:
        lines.append(f"\n{'─'*70}")
        lines.append(f"CROSS-VALIDATION RESULTS  ({N_FOLDS}-fold on 85% train+val)")
        lines.append(f"{'─'*70}")
        val_accs   = [s['val_acc']   for s in fold_stats]
        train_accs = [s['train_acc'] for s in fold_stats]
        gaps       = [s['train_acc'] - s['val_acc'] for s in fold_stats]

        lines.append(f"  {'Fold':>4}  {'Train Acc':>10}  {'Val Acc':>9}  {'Gap':>7}  "
                     f"{'Best Val Loss':>14}  {'Epochs':>7}  {'Best Epoch':>10}")
        lines.append("  " + "-" * 66)
        for s in fold_stats:
            lines.append(
                f"  {s['fold']:>4}  {s['train_acc']:>10.4f}  {s['val_acc']:>9.4f}  "
                f"{s['train_acc']-s['val_acc']:>7.4f}  {s['best_val_loss']:>14.4f}  "
                f"{s['epochs_run']:>7}  {s['best_epoch']:>10}"
            )
        lines.append("")
        lines.append(f"  CV Val Acc   : mean={np.mean(val_accs):.4f}  "
                     f"std={np.std(val_accs):.4f}  "
                     f"min={np.min(val_accs):.4f}  max={np.max(val_accs):.4f}")
        lines.append(f"  CV Train Acc : mean={np.mean(train_accs):.4f}  "
                     f"std={np.std(train_accs):.4f}")

        mean_gap = np.mean(gaps)
        lines.append(f"\n{'─'*70}")
        lines.append("GENERALIZATION GAP  (train acc − val acc per fold)")
        lines.append(f"{'─'*70}")
        lines.append(f"  Mean gap : {mean_gap:.4f}")
        lines.append(f"  Std  gap : {np.std(gaps):.4f}")
        lines.append(f"  Max  gap : {np.max(gaps):.4f}  (fold {fold_stats[int(np.argmax(gaps))]['fold']})")
        lines.append(f"  Min  gap : {np.min(gaps):.4f}  (fold {fold_stats[int(np.argmin(gaps))]['fold']})")
        if mean_gap < 0.03:
            lines.append("  Interpretation: Low gap — model generalizes well.")
        elif mean_gap < 0.08:
            lines.append("  Interpretation: Moderate gap — slight overfitting.")
        else:
            lines.append("  Interpretation: Large gap — consider more regularization.")

    # ── Loss curve summary ────────────────────────────────────────────────────
    lines.append(f"\n{'─'*70}")
    lines.append("LOSS CURVE SUMMARY  (best fold)")
    lines.append(f"{'─'*70}")
    tr_losses = history['train_loss']
    va_losses = history['val_loss']
    va_accs   = history['val_acc']
    epochs_run = len(tr_losses)

    lines.append(f"  Epochs run          : {epochs_run}")
    lines.append(f"  Final train loss    : {tr_losses[-1]:.4f}")

    if va_losses:
        lines.append(f"  Best val loss       : {history['best_val_loss']:.4f}  (epoch {history['best_epoch']})")
        lines.append(f"  Best val acc        : {max(va_accs):.4f}  (epoch {int(np.argmax(va_accs))+1})")
        lines.append(f"  Final val loss      : {va_losses[-1]:.4f}")
        lines.append(f"  Final loss gap      : {va_losses[-1] - tr_losses[-1]:.4f}")

        if len(va_losses) >= 10:
            end_trend = va_losses[-1] - va_losses[-10]
            trend_str = (f"+{end_trend:.4f} (increasing — possible overfit)"
                         if end_trend > 0.01 else f"{end_trend:.4f} (stable)")
            lines.append(f"  Val loss trend (last 10 epochs): {trend_str}")

        best_ep = history['best_epoch'] - 1
        if 0 <= best_ep < len(tr_losses):
            gen_gap = va_losses[best_ep] - tr_losses[best_ep]
            lines.append(f"  Gen gap at best epoch : {gen_gap:.4f}")
            if gen_gap < 0.05:
                lines.append("  Interpretation: Low gap — good generalization.")
            elif gen_gap < 0.15:
                lines.append("  Interpretation: Moderate gap — slight overfitting.")
            else:
                lines.append("  Interpretation: Large gap — consider more regularization.")
    else:
        lines.append("  (Final model — trained for fixed epochs, no val set)")

    lines.append(f"\n{sep}")
    return lines


def print_and_save_evaluation(y_true, y_pred, letter_classes,
                               model_name: str, model_slug: str,
                               config_str: str, history: dict,
                               fold_stats: list,
                               n_trainval: int, n_test: int,
                               out_dir: str):
    lines = build_eval_lines(y_true, y_pred, letter_classes,
                             model_name, config_str, history, fold_stats,
                             n_trainval, n_test)
    for line in lines:
        print(line)

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(out_dir, f'm2_{model_slug}_eval_{ts}.txt')
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nEvaluation saved to: {out}")


def save_results_csv(y_true, y_pred, all_probs,
                     letter_classes, model_slug, out_dir):
    rows = []
    for i in range(len(y_true)):
        row = {
            'sample_idx'  : i,
            'true_letter' : letter_classes[y_true[i]],
            'pred_letter' : letter_classes[y_pred[i]],
        }
        for j, cls in enumerate(letter_classes):
            row[f'p_{cls}'] = round(float(all_probs[i, j]), 6)
        rows.append(row)

    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(out_dir, f'm2_results_{model_slug}_{ts}.csv')
    df.to_csv(out, index=False)
    print(f"Results CSV saved to: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Air Writing Letter Model  (m2_model_oneletter.py)")
    print("=" * 60)

    # --- input directory ---
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = input("\nEnter path to sequences directory: ").strip()

    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    # --- load data ---
    print(f"\nLoading sequences from: {data_dir}")
    strokes = load_all_sequences(data_dir)

    # --- encode labels ---
    letters = [s['letter'] for s in strokes]
    le      = LabelEncoder()
    le.fit(sorted(set(letters)))
    letter_classes = list(le.classes_)
    n_classes      = len(letter_classes)
    print(f"Letters found: {letter_classes}  ({n_classes} classes)")

    # --- split ---
    # --- 85/15 test split ---
    y_all              = le.transform(letters)
    idx                = np.arange(len(strokes))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_RATIO,
        stratify=y_all[idx], random_state=RANDOM_SEED
    )
    test_strokes = [strokes[i] for i in idx_test]
    test_ds      = LetterDataset(test_strokes, le)
    test_loader  = make_loader(test_ds, shuffle=False)
    print(f"\nTrain+Val : {len(idx_trainval)} samples  (85%)")
    print(f"Test      : {len(idx_test)} samples  (15% held-out)")

    # --- model menu ---
    print("\nSelect model:")
    for k, v in MODEL_MENU.items():
        print(f"  {k}. {v}")
    while True:
        choice = input("Enter number (1-3): ").strip()
        if choice in MODEL_MENU:
            break
        print("  Invalid. Enter 1, 2, or 3.")

    model_name = MODEL_MENU[choice]
    model_slug = MODEL_SLUGS[choice]

    # --- hyperparameters ---
    print(f"\n  [{model_name} configuration]")
    n_epochs = int(input("    Epochs        (default 500): ").strip() or 500)
    lr       = float(input("    Learning rate: ").strip())
    lam      = float(input("    Lambda (boundary loss weight): ").strip())

    # --- device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # --- 3-fold CV on trainval ---
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_trainval = y_all[idx_trainval]

    best_val_loss_cv = float('inf')
    best_model       = None
    fold_stats       = []

    print(f"\n  === {model_name} — {N_FOLDS}-fold CV ===")
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(idx_trainval, y_trainval), 1):
        print(f"\n  -- Fold {fold_i}/{N_FOLDS} --")
        tr_strokes = [strokes[idx_trainval[i]] for i in tr_idx]
        va_strokes = [strokes[idx_trainval[i]] for i in va_idx]
        print(f"  Train: {len(tr_strokes)}  Val: {len(va_strokes)}")

        train_ds     = LetterDataset(tr_strokes, le)
        val_ds       = LetterDataset(va_strokes, le)
        train_loader = make_loader(train_ds, shuffle=True)
        val_loader   = make_loader(val_ds,   shuffle=False)

        if choice == '1':
            model = GRUModel(n_classes)
        elif choice == '2':
            model = LSTMModel(n_classes)
        else:
            model = TransformerModel(n_classes)

        model, history = train_model(model, train_loader, val_loader,
                                     n_epochs, lr, lam, device,
                                     f'{model_name} fold {fold_i}')

        y_tr_true, y_tr_pred, _ = predict(model, train_loader, device)
        tr_acc        = accuracy_score(y_tr_true, y_tr_pred)
        fold_val_loss = history['best_val_loss']
        fold_val_acc  = max(history['val_acc'])

        fold_stats.append({
            'fold'         : fold_i,
            'val_acc'      : fold_val_acc,
            'train_acc'    : tr_acc,
            'best_val_loss': fold_val_loss,
            'epochs_run'   : len(history['train_loss']),
            'best_epoch'   : history['best_epoch'],
            'history'      : history,
            'n_train'      : len(tr_strokes),
            'n_val'        : len(va_strokes),
        })
        print(f"  Fold {fold_i} — best val loss: {fold_val_loss:.4f}  "
              f"val acc: {fold_val_acc:.4f}  train acc: {tr_acc:.4f}")

        if fold_val_loss < best_val_loss_cv:
            best_val_loss_cv = fold_val_loss
            best_model       = copy.deepcopy(model)
            print(f"  ** New best model (fold {fold_i}) **")

    # --- evaluate best CV model on held-out test ---
    y_true, y_pred, all_probs = predict(best_model, test_loader, device)
    config_str    = f'epochs={n_epochs}  lr={lr}  lambda={lam}'
    best_fold_idx = int(np.argmin([s['best_val_loss'] for s in fold_stats]))
    best_history  = fold_stats[best_fold_idx]['history']

    print_and_save_evaluation(
        y_true, y_pred, letter_classes,
        model_name, model_slug, config_str, best_history,
        fold_stats, len(idx_trainval), len(idx_test),
        data_dir
    )

    # --- save CV best model and results ---
    save_model(best_model, model_slug, letter_classes, data_dir)
    save_results_csv(y_true, y_pred, all_probs,
                     letter_classes, model_slug, data_dir)

    # --- optional: train final model on full 80/20 split ---
    print("\n" + "=" * 70)
    train_final = input(
        "Train a final model on 80% of all data with these hyperparameters? (y/n): "
    ).strip().lower()

    if train_final == 'y':
        print("\n  === Final model — 80/20 split ===")

        mean_best_epoch = int(round(np.mean([s['best_epoch'] for s in fold_stats])))
        print(f"  Mean best epoch across {N_FOLDS} folds: {mean_best_epoch}")
        final_epochs = int(input(f"  Epochs for final model (default {mean_best_epoch}): ").strip() or mean_best_epoch)

        idx_final_train, idx_final_test = train_test_split(
            idx, test_size=0.20,
            stratify=y_all[idx], random_state=RANDOM_SEED
        )
        final_train_strokes = [strokes[i] for i in idx_final_train]
        final_test_strokes  = [strokes[i] for i in idx_final_test]
        print(f"  Training final model for {final_epochs} epochs on 80% data.")
        print(f"  Train: {len(final_train_strokes)}  Test: {len(final_test_strokes)}")

        final_train_ds     = LetterDataset(final_train_strokes, le)
        final_test_ds      = LetterDataset(final_test_strokes,  le)
        final_train_loader = make_loader(final_train_ds, shuffle=True)
        final_test_loader  = make_loader(final_test_ds,  shuffle=False)

        if choice == '1':
            final_model = GRUModel(n_classes)
        elif choice == '2':
            final_model = LSTMModel(n_classes)
        else:
            final_model = TransformerModel(n_classes)

        final_model, final_history = train_model_fixed_epochs(
            final_model, final_train_loader,
            final_epochs, lr, lam, device, model_name
        )

        yf_true, yf_pred, yf_probs = predict(final_model, final_test_loader, device)
        final_config = (f'epochs={final_epochs}  lr={lr}  lambda={lam}  split=80/20_final')

        print_and_save_evaluation(
            yf_true, yf_pred, letter_classes,
            f'{model_name} (final)', f'{model_slug}_final',
            final_config, final_history,
            [], len(final_train_strokes), len(final_test_strokes),
            data_dir
        )

        save_model(final_model, f'{model_slug}_final', letter_classes, data_dir)
        save_results_csv(yf_true, yf_pred, yf_probs,
                         letter_classes, f'{model_slug}_final', data_dir)
        print("\n  Final model training complete.")


if __name__ == '__main__':
    main()
