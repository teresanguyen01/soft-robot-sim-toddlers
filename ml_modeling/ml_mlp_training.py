#!/usr/bin/env python3
import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback if tqdm not installed

# ---------------------- Utils ----------------------

TIME_COLS_LOWER = {"time_ms", "timestamp", "frame"}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_numeric_df(path: str) -> pd.DataFrame:
    """Read CSV with headers, drop time/index cols, coerce to numeric, drop all-NaN cols."""
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    drop_cols = [lower[c] for c in lower if c in TIME_COLS_LOWER]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df

def align_by_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex test to train's column order; keep train columns only."""
    cols = list(train_df.columns)
    return train_df.copy(), test_df.reindex(columns=cols)

def drop_rowwise_nans(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X.loc[mask].reset_index(drop=True), Y.loc[mask].reset_index(drop=True)

def standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray,
                           Ytr: np.ndarray, Yte: np.ndarray):
    sx, sy = StandardScaler(), StandardScaler()
    Xtr_s = sx.fit_transform(Xtr)
    Xte_s = sx.transform(Xte)
    Ytr_s = sy.fit_transform(Ytr)
    Yte_s = sy.transform(Yte)
    return Xtr_s, Xte_s, Ytr_s, Yte_s, sx, sy

def ridge_r2(Xtr_s, Ytr_s, Xte_s, Yte_s) -> Tuple[float, float]:
    model = Ridge(alpha=1.0)
    model.fit(Xtr_s, Ytr_s)
    return model.score(Xtr_s, Ytr_s), model.score(Xte_s, Yte_s)

def apply_lag_np(X: np.ndarray, Y: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Positive lag means shift Y forward (Y[t+lag] aligned to X[t])."""
    if lag > 0:
        if lag >= len(X): raise ValueError("Lag too large for dataset length.")
        return X[:-lag], Y[lag:]
    elif lag < 0:
        k = -lag
        if k >= len(X): raise ValueError("Lag too large for dataset length.")
        return X[k:], Y[:-k]
    else:
        return X, Y

def auto_find_lag(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray,
                  lag_min: int, lag_max: int) -> Tuple[int, float, float]:
    """Return (best_lag, ridge_train_r2_at_best, ridge_test_r2_at_best)."""
    best = (0, -np.inf, -np.inf)
    for L in range(lag_min, lag_max + 1):
        # apply lag to both train/test
        try:
            XtrL, YtrL = apply_lag_np(Xtr, Ytr, L)
            XteL, YteL = apply_lag_np(Xte, Yte, L)
        except ValueError:
            continue
        if min(len(XtrL), len(XteL)) < 10:
            continue
        Xtr_s, Xte_s, Ytr_s, Yte_s, _, _ = standardize_train_test(XtrL, XteL, YtrL, YteL)
        r2_tr, r2_te = ridge_r2(Xtr_s, Ytr_s, Xte_s, Yte_s)
        if r2_te > best[2]:
            best = (L, r2_tr, r2_te)
    return best

# ---------------------- Model ----------------------

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_size),
        )
    def forward(self, x): return self.net(x)

@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.1
    patience: int = 50

def train_mlp(Xtr_s: np.ndarray, Ytr_s: np.ndarray, cfg: TrainConfig, device: torch.device) -> Tuple[MLP, dict]:
    X = torch.from_numpy(Xtr_s).float().to(device)
    Y = torch.from_numpy(Ytr_s).float().to(device)
    n = X.size(0)
    n_val = max(1, int(cfg.val_frac * n))
    X_val, Y_val = X[-n_val:], Y[-n_val:]
    X_tr,  Y_tr  = X[:-n_val], Y[:-n_val]
    if len(X_tr) == 0:  # in case tiny dataset
        X_tr, Y_tr, X_val, Y_val = X, Y, X, Y

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=cfg.batch_size, shuffle=True)

    model = MLP(X.shape[1], Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss()
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=False)

    best_val = np.inf
    bad = 0
    best_state = None

    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val), Y_val).item()
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val}

def evaluate(model: nn.Module, X_s: np.ndarray, Y_s: np.ndarray, scaler_y: StandardScaler) -> dict:
    model.eval()
    device = next(model.parameters()).device  # model's device (cuda or cpu)
    with torch.no_grad():
        X_t = torch.from_numpy(X_s).float().to(device)
        Yhat_s = model(X_t).detach().cpu().numpy()  # bring back to CPU for sklearn
    r2_scaled = r2_score(Y_s, Yhat_s)
    mse_scaled = mean_squared_error(Y_s, Yhat_s)

    # original units
    Y_true = scaler_y.inverse_transform(Y_s)
    Y_pred = scaler_y.inverse_transform(Yhat_s)
    r2_real = r2_score(Y_true, Y_pred)
    mse_real = mean_squared_error(Y_true, Y_pred)

    return {
        "r2_scaled": r2_scaled, "mse_scaled": mse_scaled,
        "r2_real": r2_real, "mse_real": mse_real
    }


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Sensor→MoCap MLP with column alignment and optional lag auto-sweep.")
    ap.add_argument("--train_input", required=True, help="CSV: training sensor (X)")
    ap.add_argument("--train_output", required=True, help="CSV: training mocap (Y)")
    ap.add_argument("--test_input", required=True, help="CSV: testing sensor (X)")
    ap.add_argument("--test_output", required=True, help="CSV: testing mocap (Y)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=50)

    ap.add_argument("--lag", type=int, default=None, help="Fixed lag (frames). If set, skip auto sweep.")
    ap.add_argument("--lag_min", type=int, default=-30, help="Auto-sweep min lag (inclusive).")
    ap.add_argument("--lag_max", type=int, default=30, help="Auto-sweep max lag (inclusive).")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load
    Xtr_df = load_numeric_df(args.train_input)
    Ytr_df = load_numeric_df(args.train_output)
    Xte_df = load_numeric_df(args.test_input)
    Yte_df = load_numeric_df(args.test_output)

    # 2) Align columns by name
    Xtr_df, Xte_df = align_by_columns(Xtr_df, Xte_df)
    Ytr_df, Yte_df = align_by_columns(Ytr_df, Yte_df)

    # 3) Drop rows with any NaNs (train/test separately)
    Xtr_df, Ytr_df = drop_rowwise_nans(Xtr_df, Ytr_df)
    Xte_df, Yte_df = drop_rowwise_nans(Xte_df, Yte_df)

    print(f"Shapes (before lag): Xtr={Xtr_df.shape}, Ytr={Ytr_df.shape}, Xte={Xte_df.shape}, Yte={Yte_df.shape}")
    print("Columns aligned?",
          Xtr_df.columns.tolist() == Xte_df.columns.tolist(),
          Ytr_df.columns.tolist() == Yte_df.columns.tolist())

    # 4) Numpy
    Xtr = Xtr_df.to_numpy(dtype=np.float64)
    Ytr = Ytr_df.to_numpy(dtype=np.float64)
    Xte = Xte_df.to_numpy(dtype=np.float64)
    Yte = Yte_df.to_numpy(dtype=np.float64)

    # 5) Optional lag auto-sweep (Ridge baseline)
    if args.lag is None:
        print(f"Auto-sweeping lag in [{args.lag_min}, {args.lag_max}] (frames) with Ridge baseline…")
        best_lag, r2tr_b, r2te_b = auto_find_lag(Xtr, Ytr, Xte, Yte, args.lag_min, args.lag_max)
        print(f"Best lag={best_lag}  Ridge R2 train/test = {r2tr_b:.3f}/{r2te_b:.3f}")
        lag = best_lag
    else:
        lag = int(args.lag)
        print(f"Using fixed lag={lag} (skipping auto-sweep)")

    # 6) Apply lag
    XtrL, YtrL = apply_lag_np(Xtr, Ytr, lag)
    XteL, YteL = apply_lag_np(Xte, Yte, lag)
    print(f"Shapes (after lag): Xtr={XtrL.shape}, Ytr={YtrL.shape}, Xte={XteL.shape}, Yte={YteL.shape}")

    # 7) Scale (fit on lagged train, apply to lagged test)
    Xtr_s, Xte_s, Ytr_s, Yte_s, sx, sy = standardize_train_test(XtrL, XteL, YtrL, YteL)

    # 8) Baseline after lag (should be > before if lag helped)
    r2tr_lin, r2te_lin = ridge_r2(Xtr_s, Ytr_s, Xte_s, Yte_s)
    print(f"Ridge R2 after lag  train/test = {r2tr_lin:.3f}/{r2te_lin:.3f}")

    # 9) Train MLP (Adam, val split, early stopping)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, val_frac=args.val_frac, patience=args.patience)
    model, info = train_mlp(Xtr_s, Ytr_s, cfg, device=device)
    print(f"Best val loss: {info['best_val_loss']:.4f}")

    # 10) Eval (scaled & original units)
    train_metrics = evaluate(model, Xtr_s, Ytr_s, scaler_y=sy)
    test_metrics  = evaluate(model, Xte_s, Yte_s, scaler_y=sy)

    print("\n=== MLP Metrics (scaled space) ===")
    print(f"Train R2: {train_metrics['r2_scaled']:.3f}   Train MSE: {train_metrics['mse_scaled']:.4f}")
    print(f"Test  R2: {test_metrics['r2_scaled']:.3f}   Test  MSE: {test_metrics['mse_scaled']:.4f}")

    print("\n=== MLP Metrics (original units) ===")
    print(f"Train R2: {train_metrics['r2_real']:.3f}   Train MSE: {train_metrics['mse_real']:.4f}")
    print(f"Test  R2: {test_metrics['r2_real']:.3f}   Test  MSE: {test_metrics['mse_real']:.4f}")

    # 11) Save model weights & scalers
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/best_mlp_state_dict.pt")  # weights only
    np.savez("artifacts/scalers.npz",
             sx_mean=sx.mean_, sx_scale=sx.scale_,
             sy_mean=sy.mean_, sy_scale=sy.scale_)
    with open("artifacts/columns.txt", "w") as f:
        f.write("X_cols:\n")
        for c in Xtr_df.columns: f.write(f"{c}\n")
        f.write("\nY_cols:\n")
        for c in Ytr_df.columns: f.write(f"{c}\n")
    print("\nSaved: artifacts/best_mlp_state_dict.pt, artifacts/scalers.npz, artifacts/columns.txt")
    print(f"Final lag used: {lag} (frames)")

if __name__ == "__main__":
    main()
