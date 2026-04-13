"""
Air Writing Model Training & Evaluation  (m1_model.py)
-------------------------------------------------------
Input  : m1_dataset_{timestamp}.npz
Output : m1_results_{modelname}_{timestamp}.csv
         m1_{modelname}_dir_{timestamp}.pt / .pkl  (saved direction model)
         m1_{modelname}_eval_{timestamp}.txt        (evaluation report)

Models
------
1. 1D CNN (raw)           X_raw         (n, 4, 25)  PyTorch
2. 1D CNN (features)      X_feat_2d     (n, 4, 12)  PyTorch
3. SVM linear             X_feat_flat   (n, 48)      sklearn
4. SVM RBF                X_feat_flat   (n, 48)      sklearn
5. Logistic Regression    X_feat_flat   (n, 48)      sklearn
6. Shallow MLP            X_feat_flat   (n, 48)      sklearn
7. Random Forest          X_feat_flat   (n, 48)      sklearn
8. 1D CNN (subwindow)     X_feat_subwin (n, 48, 11)  PyTorch

Training
--------
- 85% train+val / 15% held-out test split (stratified on direction)
- 6-fold cross-validation on the 85% train+val portion
- Best fold model (lowest val loss for CNN, highest val acc for sklearn)
  is evaluated on the held-out test set and saved to disk

Each model predicts direction (9 classes) only.
Speed head has been removed.
"""

import os
import sys
import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED         = 42
TEST_RATIO          = 0.15    # held-out test set
N_FOLDS             = 6       # k-fold CV on remaining 85%
EARLY_STOP_PATIENCE  = 10
RBF_MAX_TRAIN        = 1800   # max training samples for RBF SVM per fold (~200 per class)

DIR_CLASSES = ['E', 'N', 'NE', 'NW', 'REST', 'S', 'SE', 'SW', 'W']

MODEL_MENU = {
    '1': '1D CNN (raw)',
    '2': '1D CNN (features)',
    '3': 'SVM linear',
    '4': 'SVM RBF',
    '5': 'Logistic Regression',
    '6': 'Shallow MLP',
    '7': 'Random Forest',
    '8': '1D CNN (subwindow features)',
}

MODEL_SLUGS = {
    '1': 'cnn_raw',
    '2': 'cnn_feat',
    '3': 'svm_linear',
    '4': 'svm_rbf',
    '5': 'logistic_regression',
    '6': 'mlp',
    '7': 'random_forest',
    '8': 'cnn_subwin',
}


# =============================================================================
# 1D CNN ARCHITECTURE  (StretchNet1D)
# =============================================================================

class StretchNet1D(nn.Module):
    def __init__(self, n_channels=4, n_classes=9, n_timepoints=25,
                 filters=(16, 32), kernel_size=5, dropout_p=0.3):
        super().__init__()
        f1, f2 = filters
        pad = kernel_size // 2
        self.conv1   = nn.Conv1d(n_channels, f1, kernel_size=kernel_size, padding=pad)
        self.bn1     = nn.BatchNorm1d(f1)
        self.conv2   = nn.Conv1d(f1, f2, kernel_size=kernel_size, padding=pad)
        self.bn2     = nn.BatchNorm1d(f2)
        self.pool    = nn.MaxPool1d(kernel_size=2)
        time_after_pools = n_timepoints // 4
        fc_input_size    = f2 * time_after_pools
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1     = nn.Linear(fc_input_size, 64)
        self.fc2     = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# =============================================================================
# SUBWINDOW CNN ARCHITECTURE  (SubwinNet1D)
# =============================================================================

class SubwinNet1D(nn.Module):
    def __init__(self, n_classes=9, n_filters=16, kernel_size=2):
        super().__init__()
        pad              = kernel_size // 2
        # output length after conv with padding: 11 + 2*pad - kernel_size + 1
        out_len          = 11 + 2 * pad - kernel_size + 1
        self.conv        = nn.Conv1d(48, n_filters, kernel_size=kernel_size, padding=pad)
        self.bn          = nn.BatchNorm1d(n_filters)
        self.dropout     = nn.Dropout(p=0.3)
        self.fc1         = nn.Linear(n_filters * out_len, 64)
        self.fc2         = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
# =============================================================================

def load_npz(path: str):
    data          = np.load(path, allow_pickle=True)
    X_raw         = data['X_raw'].astype(np.float32)          # (n, 25, 4)
    X_feat_2d     = data['X_feat_2d'].astype(np.float32)      # (n, 12, 4)
    X_feat_flat   = data['X_feat_flat'].astype(np.float32)    # (n, 48)
    X_feat_subwin = data['X_feat_subwin'].astype(np.float32)  # (n, 48, 11)
    y_direction   = data['y_direction'].astype(np.int64)
    dir_classes   = list(data['dir_classes'])
    return X_raw, X_feat_2d, X_feat_flat, X_feat_subwin, y_direction, dir_classes


# =============================================================================
# DATA SPLITTING — 85/15 TEST SPLIT + 6-FOLD CV
# =============================================================================

def make_subset(X_raw, X_feat_2d, X_feat_flat, X_feat_subwin, y_direction, idx):
    return {
        'X_raw'        : X_raw[idx],
        'X_feat_2d'    : X_feat_2d[idx],
        'X_feat_flat'  : X_feat_flat[idx],
        'X_feat_subwin': X_feat_subwin[idx],
        'y_dir'        : y_direction[idx],
    }


def split_test(X_raw, X_feat_2d, X_feat_flat, X_feat_subwin, y_direction):
    """
    Stratified 85/15 split. Returns (trainval_data, test_data).
    """
    idx = np.arange(len(y_direction))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_RATIO, stratify=y_direction, random_state=RANDOM_SEED
    )
    trainval = make_subset(X_raw, X_feat_2d, X_feat_flat, X_feat_subwin,
                           y_direction, idx_trainval)
    test     = make_subset(X_raw, X_feat_2d, X_feat_flat, X_feat_subwin,
                           y_direction, idx_test)

    print(f"  Train+Val : {len(idx_trainval)} samples  (85%)")
    print(f"  Test      : {len(idx_test)} samples  (15% held-out)")
    return trainval, test


def get_cv_folds(trainval_data):
    """
    Returns list of (train_subset, val_subset) dicts for 6-fold CV
    stratified on direction labels.
    """
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_dir = trainval_data['y_dir']
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(y_dir, y_dir)):
        tr = {k: v[tr_idx] for k, v in trainval_data.items()}
        va = {k: v[va_idx] for k, v in trainval_data.items()}
        folds.append((tr, va))
        print(f"  Fold {fold_idx+1}: train={len(tr_idx)}  val={len(va_idx)}")
    return folds


# =============================================================================
# CNN TRAINING HELPERS
# =============================================================================

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_cnn(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int,
              lr: float,
              device: torch.device,
              label: str = '') -> nn.Module:
    """
    Train a StretchNet1D model with early stopping on val loss.
    Prints train loss, val loss, val accuracy per epoch.
    Returns best model (lowest val loss).
    """
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    criterion   = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())
    patience_ctr  = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    model.to(device)

    print(f"\n  Training CNN [{label}] for up to {n_epochs} epochs, lr={lr}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*8}")

    for epoch in range(1, n_epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # --- val ---
        model.eval()
        val_loss   = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb  = xb.to(device), yb.to(device)
                logits   = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_acc:>8.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


def predict_cnn(model: nn.Module,
                X: np.ndarray,
                device: torch.device) -> tuple:
    """Returns (predicted class indices, probability arrays)."""
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    probs  = np.concatenate(all_probs, axis=0)
    preds  = probs.argmax(axis=1)
    return preds, probs


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_model(model, model_type: str, tag: str, out_dir: str) -> str:
    """Save CNN (.pt) or sklearn (.pkl) model. Returns saved path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if model_type == 'cnn':
        path = os.path.join(out_dir, f'm1_{tag}_{ts}.pt')
        torch.save(model.state_dict(), path)
    else:
        path = os.path.join(out_dir, f'm1_{tag}_{ts}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    print(f"  Model saved: {path}")
    return path


# =============================================================================
# CNN PIPELINE  (6-fold CV)
# =============================================================================

def run_cnn(trainval, test, mode: str, dir_classes,
            model_slug: str, out_dir: str):
    """
    mode: 'raw' or 'feat'
    Runs 6-fold CV on direction only, saves best model, evaluates on test.
    For mode='raw', prompts for filter sizes, kernel size, and dropout rate.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    def get_X(data):
        if mode == 'raw':
            return data['X_raw'].transpose(0, 2, 1)
        else:
            return data['X_feat_2d'].transpose(0, 2, 1)

    n_timepoints = 25 if mode == 'raw' else 12
    n_dir = len(dir_classes)

    print(f"\n  --- {'Raw CNN' if mode == 'raw' else 'Feature CNN'} Configuration ---")
    dir_epochs = int(input("\n    Epochs (default 500): ").strip() or 500)
    dir_lr     = float(input("    Learning rate: ").strip())

    if mode == 'raw':
        print("\n    Filter sizes (e.g. 1=16,32  2=8,16):")
        print("      1. 16, 32  (default)")
        print("      2.  8, 16")
        filt_choice = input("    Choose (1-2, default 1): ").strip() or '1'
        filters     = (16, 32) if filt_choice != '2' else (8, 16)

        print("\n    Kernel size:")
        print("      1. 5  (default)")
        print("      2. 3")
        kern_choice = input("    Choose (1-2, default 1): ").strip() or '1'
        kernel_size = 5 if kern_choice != '2' else 3

        print("\n    Dropout rate:")
        print("      1. 0.3  (default)")
        print("      2. 0.5")
        drop_choice = input("    Choose (1-2, default 1): ").strip() or '1'
        dropout_p   = 0.3 if drop_choice != '2' else 0.5

        print(f"\n    Config: filters={filters}  kernel={kernel_size}  dropout={dropout_p}")
    else:
        filters     = (16, 32)
        kernel_size = 5
        dropout_p   = 0.3

    folds = get_cv_folds(trainval)

    print("\n  === Direction model — 6-fold CV ===")
    best_val_loss = float('inf')
    best_model    = None
    fold_stats    = []
    for fold_i, (tr, va) in enumerate(folds, 1):
        print(f"\n  -- Fold {fold_i}/{N_FOLDS} --")
        model = StretchNet1D(n_channels=4, n_classes=n_dir, n_timepoints=n_timepoints,
                             filters=filters, kernel_size=kernel_size, dropout_p=dropout_p)
        model, history = train_cnn(model,
                          make_loader(get_X(tr), tr['y_dir']),
                          make_loader(get_X(va), va['y_dir'], shuffle=False),
                          n_epochs=dir_epochs, lr=dir_lr,
                          device=device, label=f'fold {fold_i}')
        val_loss = history['val_loss'][-1]
        val_acc  = history['val_acc'][-1]
        # compute train acc on fold training set
        tr_preds, _ = predict_cnn(model, get_X(tr), device)
        tr_acc = accuracy_score(tr['y_dir'], tr_preds)
        fold_stats.append({
            'fold': fold_i, 'val_acc': val_acc, 'train_acc': tr_acc,
            'best_val_loss': min(history['val_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': val_loss,
            'epochs_run': len(history['train_loss']),
            'min_val_loss_epoch': int(np.argmin(history['val_loss'])) + 1,
            'history': history,
        })
        print(f"  Fold {fold_i} val loss: {val_loss:.4f}  val acc: {val_acc:.4f}  train acc: {tr_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model    = copy.deepcopy(model)
            print(f"  ** New best model (fold {fold_i}) **")

    save_model(best_model, 'cnn', f'{model_slug}_dir', out_dir)
    dir_preds_test, dir_probs_test = predict_cnn(best_model, get_X(test), device)
    config_str = f'filters={filters}  kernel={kernel_size}  dropout={dropout_p}  lr={dir_lr}' if mode == 'raw' else f'lr={dir_lr}'
    return dir_preds_test, dir_probs_test, test['y_dir'], config_str, fold_stats


def run_subwin_cnn(trainval, test, dir_classes,
                   model_slug: str, out_dir: str):
    """SubwinNet1D on X_feat_subwin (n, 48, 11) — 6-fold CV, direction only."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    n_dir = len(dir_classes)

    print("\n  --- SubwinNet1D Configuration ---")
    print("\n    Learning rate:")
    print("      1. 0.001   (default)")
    print("      2. 0.0001")
    lr_choice  = input("    Choose (1-2, default 1): ").strip() or '1'
    dir_lr     = 0.001 if lr_choice != '2' else 0.0001

    print("\n    Number of filters:")
    print("      1. 16  (default)")
    print("      2.  8")
    print("      3. 32")
    filt_choice = input("    Choose (1-3, default 1): ").strip() or '1'
    n_filters   = {'1': 16, '2': 8, '3': 32}.get(filt_choice, 16)

    print("\n    Kernel size:")
    print("      1. 2  (default)")
    print("      2. 4")
    kern_choice = input("    Choose (1-2, default 1): ").strip() or '1'
    kernel_size = 2 if kern_choice != '2' else 4

    dir_epochs = int(input("\n    Epochs (default 500): ").strip() or 500)

    print(f"\n    Config: lr={dir_lr}  filters={n_filters}  kernel={kernel_size}")

    folds = get_cv_folds(trainval)

    print("\n  === Direction model — 6-fold CV ===")
    best_val_loss = float('inf')
    best_model    = None
    fold_stats    = []
    for fold_i, (tr, va) in enumerate(folds, 1):
        print(f"\n  -- Fold {fold_i}/{N_FOLDS} --")
        model = SubwinNet1D(n_classes=n_dir, n_filters=n_filters, kernel_size=kernel_size)
        model, history = train_cnn(model,
                          make_loader(tr['X_feat_subwin'], tr['y_dir']),
                          make_loader(va['X_feat_subwin'], va['y_dir'], shuffle=False),
                          n_epochs=dir_epochs, lr=dir_lr,
                          device=device, label=f'fold {fold_i}')
        val_loss = history['val_loss'][-1]
        val_acc  = history['val_acc'][-1]
        tr_preds, _ = predict_cnn(model, tr['X_feat_subwin'], device)
        tr_acc = accuracy_score(tr['y_dir'], tr_preds)
        fold_stats.append({
            'fold': fold_i, 'val_acc': val_acc, 'train_acc': tr_acc,
            'best_val_loss': min(history['val_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': val_loss,
            'epochs_run': len(history['train_loss']),
            'min_val_loss_epoch': int(np.argmin(history['val_loss'])) + 1,
            'history': history,
        })
        print(f"  Fold {fold_i} val loss: {val_loss:.4f}  val acc: {val_acc:.4f}  train acc: {tr_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model    = copy.deepcopy(model)
            print(f"  ** New best model (fold {fold_i}) **")

    save_model(best_model, 'cnn', f'{model_slug}_dir', out_dir)
    dir_preds_test, dir_probs_test = predict_cnn(best_model, test['X_feat_subwin'], device)
    config_str = f'lr={dir_lr}  filters={n_filters}  kernel={kernel_size}'
    return dir_preds_test, dir_probs_test, test['y_dir'], config_str, fold_stats


# =============================================================================
# SKLEARN PIPELINE  (6-fold CV)
# =============================================================================

def get_sklearn_model(choice: str, C: float = 1.0, gamma: float = 'scale',
                      n_estimators: int = 200, max_features: int = 7,
                      max_depth=None, hidden_layer_sizes=(128, 64),
                      alpha: float = 0.0001):
    if choice == '3':
        base  = LinearSVC(C=C, max_iter=2000, random_state=RANDOM_SEED)
        model = CalibratedClassifierCV(base, cv=3)
    elif choice == '4':
        model = CalibratedClassifierCV(
                    SVC(kernel='rbf', C=C, gamma=gamma, random_state=RANDOM_SEED),
                    cv=3)
    elif choice == '5':
        model = LogisticRegression(C=C, max_iter=5000,
                                   solver='lbfgs', random_state=RANDOM_SEED)
    elif choice == '6':
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              activation='relu', alpha=alpha,
                              max_iter=500, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=10,
                              random_state=RANDOM_SEED)
    elif choice == '7':
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_features=max_features,
                                       max_depth=max_depth,
                                       random_state=RANDOM_SEED)
    # wrap in pipeline with StandardScaler so features are scaled per fold
    return Pipeline([('scaler', StandardScaler()), ('model', model)])


def prompt_svm_hyperparams(choice: str) -> tuple:
    """
    Prompt for SVM/LR hyperparameters as free-form numeric input.
    Returns (C, gamma, config_str).
    """
    print("\n  --- Model Configuration ---")

    if choice == '5':
        while True:
            try:
                lam = float(input("\n    λ (regularization, e.g. 0.1, 1, 10): ").strip())
                break
            except ValueError:
                print("    Invalid. Enter a numeric value.")
        C = 1.0 / lam
        config_str = f'lambda={lam}  (C={C:.4f})'
        print(f"    Config: {config_str}")
        return C, 'scale', config_str

    while True:
        try:
            C = float(input("\n    C (regularization, e.g. 0.1, 1, 10): ").strip())
            break
        except ValueError:
            print("    Invalid. Enter a numeric value.")

    gamma = 'scale'
    if choice == '4':
        val = input("    Gamma (e.g. scale, 0.001, 0.02, 0.1) [default: scale]: ").strip()
        if val == '' or val.lower() == 'scale':
            gamma = 'scale'
        else:
            while True:
                try:
                    gamma = float(val)
                    break
                except ValueError:
                    val = input("    Invalid. Enter a numeric value or 'scale': ").strip()
                    if val.lower() == 'scale':
                        gamma = 'scale'
                        break

    config_str = f'C={C}' + (f'  gamma={gamma}' if choice == '4' else '')
    print(f"    Config: {config_str}")
    return C, gamma, config_str


def prompt_mlp_hyperparams() -> tuple:
    """Prompt for MLP hyperparameters. Returns (hidden_layer_sizes, alpha, config_str)."""
    print("\n  --- Shallow MLP Configuration ---")

    print("\n    Hidden layer sizes:")
    print("      1. (128, 64)  (default)")
    print("      2. (64, 32)")
    layer_choice       = input("    Choose (1-2, default 1): ").strip() or '1'
    hidden_layer_sizes = (128, 64) if layer_choice != '2' else (64, 32)

    while True:
        try:
            alpha = float(input("\n    Alpha / L2 regularization (e.g. 0.0001, 0.001, 0.01): ").strip())
            break
        except ValueError:
            print("    Invalid. Enter a numeric value.")

    config_str = f'hidden={hidden_layer_sizes}  alpha={alpha}'
    print(f"    Config: {config_str}")
    return hidden_layer_sizes, alpha, config_str


def prompt_rf_hyperparams() -> tuple:
    """Prompt for Random Forest hyperparameters. Returns (n_estimators, max_features, max_depth, config_str)."""
    print("\n  --- Random Forest Configuration ---")

    while True:
        try:
            n_estimators = int(input("\n    Number of trees (e.g. 100, 200): ").strip())
            break
        except ValueError:
            print("    Invalid. Enter an integer.")

    while True:
        try:
            max_features = int(input("    Max features per split (e.g. 5, 7): ").strip())
            break
        except ValueError:
            print("    Invalid. Enter an integer.")

    depth_val = input("    Max depth (e.g. 5, 10, or 'none' for unlimited) [default: none]: ").strip().lower()
    max_depth = None if depth_val in ('', 'none') else int(depth_val)

    config_str = f'trees={n_estimators}  max_features={max_features}  max_depth={max_depth}'
    print(f"    Config: {config_str}")
    return n_estimators, max_features, max_depth, config_str


def run_sklearn(trainval, test, choice: str, dir_classes,
                model_slug: str, out_dir: str):
    """6-fold CV on trainval, direction only. Best fold saved and evaluated."""
    if choice in ('3', '4', '5'):
        C, gamma, config_str = prompt_svm_hyperparams(choice)
        n_estimators, max_features, max_depth = 200, 7, None
        hidden_layer_sizes, alpha = (128, 64), 0.0001
    elif choice == '6':
        hidden_layer_sizes, alpha, config_str = prompt_mlp_hyperparams()
        C, gamma = 1.0, 'scale'
        n_estimators, max_features, max_depth = 200, 7, None
    elif choice == '7':
        n_estimators, max_features, max_depth, config_str = prompt_rf_hyperparams()
        C, gamma = 1.0, 'scale'
        hidden_layer_sizes, alpha = (128, 64), 0.0001
    else:
        C, gamma, config_str = 1.0, 'scale', ''
        n_estimators, max_features, max_depth = 200, 7, None
        hidden_layer_sizes, alpha = (128, 64), 0.0001

    folds = get_cv_folds(trainval)

    print("\n  === Direction model — 6-fold CV ===")
    best_val_acc = -1.0
    best_model   = None
    fold_stats   = []
    for fold_i, (tr, va) in enumerate(folds, 1):
        X_tr = tr['X_feat_flat']
        y_tr = tr['y_dir']

        # RBF SVM is O(n²) — subsample to keep training time manageable
        if choice == '4' and len(y_tr) > RBF_MAX_TRAIN:
            rng     = np.random.default_rng(RANDOM_SEED + fold_i)
            classes = np.unique(y_tr)
            per_cls = RBF_MAX_TRAIN // len(classes)
            sub_idx = np.concatenate([
                rng.choice(np.where(y_tr == c)[0],
                           size=min(per_cls, (y_tr == c).sum()),
                           replace=False)
                for c in classes
            ])
            rng.shuffle(sub_idx)
            X_tr = X_tr[sub_idx]
            y_tr = y_tr[sub_idx]
            print(f"  Fold {fold_i}: RBF subsampled {len(y_tr)} / {len(tr['y_dir'])} samples")

        model = get_sklearn_model(choice, C=C, gamma=gamma,
                                  n_estimators=n_estimators,
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  hidden_layer_sizes=hidden_layer_sizes,
                                  alpha=alpha)
        model.fit(X_tr, y_tr)
        val_acc  = accuracy_score(va['y_dir'], model.predict(va['X_feat_flat']))
        train_acc = accuracy_score(y_tr, model.predict(X_tr))
        fold_stats.append({
            'fold': fold_i, 'val_acc': val_acc, 'train_acc': train_acc,
            'n_train': len(y_tr),
        })
        print(f"  Fold {fold_i} val acc: {val_acc:.4f}  train acc: {train_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model   = copy.deepcopy(model)
            print(f"  ** New best model (fold {fold_i}) **")

    save_model(best_model, 'sklearn', f'{model_slug}_dir', out_dir)
    dir_probs_test = best_model.predict_proba(test['X_feat_flat'])
    dir_preds_test = dir_probs_test.argmax(axis=1)
    return dir_preds_test, dir_probs_test, test['y_dir'], config_str, fold_stats


# =============================================================================
# EVALUATION, PRINTING & SAVING
# =============================================================================

def build_evaluation_lines(y_true_dir, y_pred_dir, dir_classes,
                            model_name: str, n_folds: int,
                            config_str: str = '',
                            fold_stats: list = None) -> list:
    """Build comprehensive evaluation text for printing and saving."""
    lines = []
    sep   = "=" * 70

    lines.append(sep)
    lines.append(f"EVALUATION — {model_name}  ({n_folds}-fold CV, best fold on 15% test)")
    if config_str:
        lines.append(f"Config     : {config_str}")
    lines.append(sep)

    # ── Test set accuracy ────────────────────────────────────────────────────
    acc = accuracy_score(y_true_dir, y_pred_dir)
    n   = len(y_true_dir)
    lines.append(f"\n{'─'*70}")
    lines.append("TEST SET RESULTS  (held-out 15%)")
    lines.append(f"{'─'*70}")
    lines.append(f"Overall accuracy : {acc:.4f}  ({int(acc*n)}/{n} correct)")

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_dir, y_pred_dir, labels=list(range(len(dir_classes))),
        zero_division=0
    )
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
        y_true_dir, y_pred_dir, average='macro', zero_division=0
    )
    wt_p, wt_r, wt_f1, _   = precision_recall_fscore_support(
        y_true_dir, y_pred_dir, average='weighted', zero_division=0
    )

    lines.append(f"Macro   P/R/F1   : {mac_p:.4f} / {mac_r:.4f} / {mac_f1:.4f}")
    lines.append(f"Weighted P/R/F1  : {wt_p:.4f} / {wt_r:.4f} / {wt_f1:.4f}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    lines.append(f"\n{'─'*70}")
    lines.append("PER-CLASS BREAKDOWN")
    lines.append(f"{'─'*70}")
    hdr = f"  {'Class':6s}  {'Acc':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>7}  {'Support':>8}"
    lines.append(hdr)
    lines.append("  " + "-" * 66)
    for i, cls in enumerate(dir_classes):
        mask    = y_true_dir == i
        if mask.sum() == 0:
            continue
        cls_acc = accuracy_score(y_true_dir[mask], y_pred_dir[mask])
        lines.append(
            f"  {cls:6s}  {cls_acc:>7.4f}  {prec[i]:>10.4f}  {rec[i]:>8.4f}  "
            f"{f1[i]:>7.4f}  {int(sup[i]):>8}"
        )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    lines.append(f"\n{'─'*70}")
    lines.append("CONFUSION MATRIX  (rows=true, cols=pred)")
    lines.append(f"{'─'*70}")
    cm     = confusion_matrix(y_true_dir, y_pred_dir)
    header = "        " + "  ".join(f"{c:>5}" for c in dir_classes)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"  {dir_classes[i]:5s}   " + "  ".join(f"{v:>5}" for v in row))

    # ── Cross-validation results ──────────────────────────────────────────────
    if fold_stats:
        lines.append(f"\n{'─'*70}")
        lines.append("CROSS-VALIDATION RESULTS  (6-fold on 85% train+val)")
        lines.append(f"{'─'*70}")

        val_accs   = [s['val_acc']   for s in fold_stats]
        train_accs = [s['train_acc'] for s in fold_stats]
        gaps       = [s['train_acc'] - s['val_acc'] for s in fold_stats]

        # per-fold table
        has_loss = 'best_val_loss' in fold_stats[0]
        if has_loss:
            lines.append(f"  {'Fold':>4}  {'Train Acc':>10}  {'Val Acc':>9}  {'Gap':>7}  "
                         f"{'Best Val Loss':>14}  {'Epochs':>7}  {'Best Epoch':>10}")
            lines.append("  " + "-" * 66)
            for s in fold_stats:
                lines.append(
                    f"  {s['fold']:>4}  {s['train_acc']:>10.4f}  {s['val_acc']:>9.4f}  "
                    f"{s['train_acc']-s['val_acc']:>7.4f}  {s['best_val_loss']:>14.4f}  "
                    f"{s['epochs_run']:>7}  {s['min_val_loss_epoch']:>10}"
                )
        else:
            lines.append(f"  {'Fold':>4}  {'Train Acc':>10}  {'Val Acc':>9}  {'Gap':>7}  {'N Train':>8}")
            lines.append("  " + "-" * 48)
            for s in fold_stats:
                lines.append(
                    f"  {s['fold']:>4}  {s['train_acc']:>10.4f}  {s['val_acc']:>9.4f}  "
                    f"{s['train_acc']-s['val_acc']:>7.4f}  {s['n_train']:>8}"
                )

        lines.append("")
        lines.append(f"  CV Val Acc   : mean={np.mean(val_accs):.4f}  "
                     f"std={np.std(val_accs):.4f}  "
                     f"min={np.min(val_accs):.4f}  max={np.max(val_accs):.4f}")
        lines.append(f"  CV Train Acc : mean={np.mean(train_accs):.4f}  "
                     f"std={np.std(train_accs):.4f}  "
                     f"min={np.min(train_accs):.4f}  max={np.max(train_accs):.4f}")

        # ── Generalization gap ───────────────────────────────────────────────
        mean_gap = np.mean(gaps)
        lines.append(f"\n{'─'*70}")
        lines.append("GENERALIZATION GAP  (train acc − val acc per fold)")
        lines.append(f"{'─'*70}")
        lines.append(f"  Mean gap : {mean_gap:.4f}")
        lines.append(f"  Std  gap : {np.std(gaps):.4f}")
        lines.append(f"  Max  gap : {np.max(gaps):.4f}  (fold {fold_stats[int(np.argmax(gaps))]['fold']})")
        lines.append(f"  Min  gap : {np.min(gaps):.4f}  (fold {fold_stats[int(np.argmin(gaps))]['fold']})")
        if mean_gap < 0.03:
            lines.append("  Interpretation: Low gap — model generalizes well, no significant overfitting.")
        elif mean_gap < 0.08:
            lines.append("  Interpretation: Moderate gap — slight overfitting, consider more regularization.")
        else:
            lines.append("  Interpretation: Large gap — model is overfitting, consider stronger regularization or less capacity.")

        # ── Loss curve summary (CNN only) ────────────────────────────────────
        if has_loss:
            lines.append(f"\n{'─'*70}")
            lines.append("LOSS CURVE SUMMARY  (CNN folds)")
            lines.append(f"{'─'*70}")
            for s in fold_stats:
                h = s['history']
                tr_losses = h['train_loss']
                va_losses = h['val_loss']
                lines.append(f"  Fold {s['fold']}:")
                lines.append(f"    Epochs run       : {s['epochs_run']}")
                lines.append(f"    Best val loss    : {s['best_val_loss']:.4f}  (epoch {s['min_val_loss_epoch']})")
                lines.append(f"    Final train loss : {tr_losses[-1]:.4f}")
                lines.append(f"    Final val loss   : {va_losses[-1]:.4f}")
                lines.append(f"    Loss gap (final) : {va_losses[-1] - tr_losses[-1]:.4f}")
                # convergence — did val loss plateau before early stopping?
                if s['epochs_run'] < s['epochs_run']:
                    lines.append(f"    Converged early  : yes")
                # monotonicity check — did val loss increase in last 10 epochs?
                if len(va_losses) >= 10:
                    end_trend = va_losses[-1] - va_losses[-10]
                    trend_str = f"+{end_trend:.4f} (increasing — overfit)" if end_trend > 0.01 else f"{end_trend:.4f} (stable)"
                    lines.append(f"    Val loss trend (last 10 epochs): {trend_str}")

    lines.append(f"\n{sep}")
    return lines


def print_and_save_evaluation(y_true_dir, y_pred_dir, dir_classes,
                               model_name: str, model_slug: str, out_dir: str,
                               config_str: str = '', fold_stats: list = None):
    lines = build_evaluation_lines(y_true_dir, y_pred_dir, dir_classes,
                                   model_name, N_FOLDS, config_str, fold_stats)
    for line in lines:
        print(line)

    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    out     = os.path.join(out_dir, f'm1_{model_slug}_eval_{ts}.txt')
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nEvaluation saved to: {out}")


# =============================================================================
# SAVE RESULTS CSV
# =============================================================================

def save_results(dir_preds, dir_probs, y_true_dir, dir_classes,
                 model_slug: str, out_dir: str):
    rows = []
    for i in range(len(y_true_dir)):
        row = {'sample_idx': i,
               'true_direction': dir_classes[y_true_dir[i]],
               'pred_direction': dir_classes[dir_preds[i]]}
        for j, cls in enumerate(dir_classes):
            row[f'p_{cls}'] = round(float(dir_probs[i, j]), 6)
        rows.append(row)

    df  = pd.DataFrame(rows)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(out_dir, f'm1_results_{model_slug}_{ts}.csv')
    df.to_csv(out, index=False)
    print(f"Results CSV saved to: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Air Writing Model  (m1_model.py)")
    print("=" * 60)

    # --- load data ---
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = input("Enter path to .npz dataset file: ").strip()

    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        sys.exit(1)

    out_dir = os.path.dirname(npz_path) or '.'

    print(f"\nLoading: {npz_path}")
    (X_raw, X_feat_2d, X_feat_flat, X_feat_subwin,
     y_direction, dir_classes) = load_npz(npz_path)
    print(f"  Samples     : {len(y_direction)}")
    print(f"  Dir classes : {dir_classes}")

    # --- 85/15 test split ---
    print("\nSplitting data (85% train+val / 15% test)...")
    trainval, test = split_test(
        X_raw, X_feat_2d, X_feat_flat, X_feat_subwin, y_direction
    )

    # --- model menu ---
    print("\nSelect model:")
    for k, v in MODEL_MENU.items():
        print(f"  {k}. {v}")
    while True:
        choice = input("Enter number (1-8): ").strip()
        if choice in MODEL_MENU:
            break
        print("  Invalid. Enter 1-8.")

    model_name = MODEL_MENU[choice]
    model_slug = MODEL_SLUGS[choice]
    print(f"\nRunning: {model_name}  ({N_FOLDS}-fold CV)")

    # --- run selected model ---
    if choice == '1':
        dir_preds, dir_probs, y_true_dir, config_str, fold_stats = run_cnn(
            trainval, test, mode='raw', dir_classes=dir_classes,
            model_slug=model_slug, out_dir=out_dir)
    elif choice == '2':
        dir_preds, dir_probs, y_true_dir, config_str, fold_stats = run_cnn(
            trainval, test, mode='feat', dir_classes=dir_classes,
            model_slug=model_slug, out_dir=out_dir)
    elif choice == '8':
        dir_preds, dir_probs, y_true_dir, config_str, fold_stats = run_subwin_cnn(
            trainval, test, dir_classes=dir_classes,
            model_slug=model_slug, out_dir=out_dir)
    else:
        dir_preds, dir_probs, y_true_dir, config_str, fold_stats = run_sklearn(
            trainval, test, choice, dir_classes=dir_classes,
            model_slug=model_slug, out_dir=out_dir)

    # --- evaluate and save ---
    print_and_save_evaluation(y_true_dir, dir_preds, dir_classes,
                               model_name, model_slug, out_dir,
                               config_str, fold_stats)
    save_results(dir_preds, dir_probs, y_true_dir, dir_classes,
                 model_slug, out_dir)


if __name__ == '__main__':
    main()
