#!/usr/bin/env python3
"""
Analyze human-joint quaternion data.

- Reads a CSV or Excel file with columns like: Frame, Pelvis q0..q3, L5 q0..q3, ...
- Renormalizes quaternions (just in case).
- Converts to Euler angles (ZYX yaw,pitch,roll) per joint.
- Plots:
  1) Time series for a small set of key joints
  2) Correlation heatmap between all Euler features
  3) PCA (first two components) over all Euler features
  4) Simple periodicity check via FFT on a chosen signal

Outputs figures into ./figs/.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for robust quaternion->Euler
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def find_joints(df_columns):
    """
    Group columns that look like '<Joint> q0..q3'.
    Returns: dict {joint_name: [q0,q1,q2,q3]}
    """
    pat = re.compile(r"^(?P<joint>.+?)\s+q([0-3])$")
    groups = {}
    for col in df_columns:
        m = pat.match(col)
        if m:
            joint = m.group("joint").strip()
            groups.setdefault(joint, []).append(col)
    # keep only complete quaternions
    groups = {j: sorted(cols, key=lambda c: int(c.split('q')[-1]))
              for j, cols in groups.items() if len(cols) == 4}
    return groups


def renorm_quats(q):
    """Renormalize quaternions to unit length."""
    # q shape: (N,4) in scalar-first (w,x,y,z) order
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return q / norms


def quat_to_euler_zyx(q):
    """
    Convert quaternion (w,x,y,z) to Euler ZYX (yaw, pitch, roll) in degrees.
    Uses SciPy if available; otherwise a minimal implementation.
    """
    if SCIPY_AVAILABLE:
        # SciPy expects (x,y,z,w)
        q_xyzw = np.column_stack([q[:,1], q[:,2], q[:,3], q[:,0]])
        return R.from_quat(q_xyzw).as_euler('zyx', degrees=True)

    # Minimal conversion (scalar-first), robust enough for unit quats
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]

    # yaw (Z)
    t0 = +2.0*(w*z + x*y)
    t1 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.degrees(np.arctan2(t0, t1))

    # pitch (Y)
    t2 = +2.0*(w*y - z*x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch = np.degrees(np.arcsin(t2))

    # roll (X)
    t3 = +2.0*(w*x + y*z)
    t4 = +1.0 - 2.0*(x*x + y*y)
    roll = np.degrees(np.arctan2(t3, t4))

    # Return columns in [yaw(Z), pitch(Y), roll(X)]
    return np.column_stack([yaw, pitch, roll])


def build_euler_dataframe(df, joint_cols, frame_col='Frame'):
    """
    For each joint, convert q0..q3 -> Euler_Z, Euler_Y, Euler_X (deg).
    Returns a dataframe with Frame + <Joint> yaw/pitch/roll columns.
    """
    out = pd.DataFrame()
    if frame_col in df.columns:
        out[frame_col] = df[frame_col].values
    else:
        out['Frame'] = np.arange(len(df))

    for joint, cols in joint_cols.items():
        q = df[cols].to_numpy(dtype=float)  # (N,4) w,x,y,z
        q = renorm_quats(q)
        eul = quat_to_euler_zyx(q)  # (N,3) yaw,pitch,roll
        out[f'{joint} yaw'] = eul[:, 0]
        out[f'{joint} pitch'] = eul[:, 1]
        out[f'{joint} roll'] = eul[:, 2]

    return out


def plot_time_series(eul_df, joints, frame_col='Frame', outdir=Path('figs')):
    outdir.mkdir(parents=True, exist_ok=True)
    for joint in joints:
        cols = [f'{joint} yaw', f'{joint} pitch', f'{joint} roll']
        if not all(c in eul_df.columns for c in cols):
            continue
        x = eul_df[frame_col].values
        y = eul_df[cols]
        plt.figure(figsize=(10, 4))
        for c in cols:
            plt.plot(x, eul_df[c], label=c)
        plt.title(f'{joint} – Euler angles (ZYX)')
        plt.xlabel('Frame')
        plt.ylabel('Degrees')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(outdir / f"ts_{joint.replace(' ', '_')}.png", dpi=200)
        plt.close()


def plot_correlation(eul_df, outdir=Path('figs')):
    outdir.mkdir(parents=True, exist_ok=True)
    feat = eul_df.drop(columns=['Frame'], errors='ignore')
    corr = feat.corr()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, interpolation='nearest', aspect='auto')
    plt.title('Feature Correlation (Euler angles)')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outdir / "correlation.png", dpi=250)
    plt.close()


def plot_pca(eul_df, n_components=2, outdir=Path('figs')):
    from sklearn.decomposition import PCA
    feat = eul_df.drop(columns=['Frame'], errors='ignore').to_numpy()
    feat = np.nan_to_num(feat, copy=False)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(feat)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    # PC time series
    for i in range(n_components):
        plt.figure(figsize=(10, 3.2))
        plt.plot(eul_df.get('Frame', pd.RangeIndex(len(eul_df))), pcs[:, i])
        plt.title(f'PCA component {i+1} (explained var: {pca.explained_variance_ratio_[i]:.2%})')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(outdir / f"pca_pc{i+1}.png", dpi=200)
        plt.close()

    # Scatter PC1 vs PC2
    if n_components >= 2:
        plt.figure(figsize=(5.5, 5.0))
        plt.scatter(pcs[:, 0], pcs[:, 1], s=6, alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA: PC1 vs PC2')
        plt.tight_layout()
        plt.savefig(outdir / "pca_scatter.png", dpi=200)
        plt.close()


def plot_periodicity(eul_df, signal_col, outdir=Path('figs')):
    """
    Very simple periodicity check: magnitude spectrum via FFT.
    Assumes uniform sampling in 'Frame' (proxy for time).
    """
    if signal_col not in eul_df.columns:
        return
    y = eul_df[signal_col].to_numpy(dtype=float)
    y = y - np.nanmean(y)
    n = len(y)
    # zero-pad to next power of two for a cleaner spectrum
    nfft = 1 << (n - 1).bit_length()
    Y = np.fft.rfft(y, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=1.0)  # "cycles per frame"
    mag = np.abs(Y)

    plt.figure(figsize=(10, 3.2))
    plt.plot(freq[1:], mag[1:])  # skip DC
    plt.title(f'FFT magnitude – {signal_col} (peaks suggest periodicity)')
    plt.xlabel('Frequency (cycles per frame)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"fft_{signal_col.replace(' ', '_')}.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Analyze joint quaternion dataset.")
    ap.add_argument("csv", help="Path to CSV or Excel file (.csv, .xlsx, .xls).")
    ap.add_argument("--frame-col", default="Frame", help="Name of frame/time column.")
    ap.add_argument("--key-joints", nargs="*", default=[
        "Pelvis", "L5", "T12", "T8", "Neck", "Head",
        "Right Shoulder", "Right Upper Arm", "Right Forearm",
        "Left Shoulder", "Left Upper Arm", "Left Forearm",
        "Right Upper Leg", "Right Lower Leg", "Right Foot",
        "Left Upper Leg", "Left Lower Leg", "Left Foot",
    ], help="Subset of joints to plot as time series.")
    ap.add_argument("--fft-signal", default="Right Foot pitch",
                    help="Column to analyze for periodicity (after Euler conversion).")
    ap.add_argument("--outdir", default="figs", help="Directory to write figures.")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # Load - handle both CSV and Excel files
    file_path = Path(args.csv)
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(args.csv)
    else:
        # Try different encodings for CSV files
        try:
            df = pd.read_csv(args.csv, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(args.csv, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(args.csv, encoding='cp1252')

    # Detect joints from headers
    joint_cols = find_joints(df.columns)
    if not joint_cols:
        raise SystemExit("No quaternion columns found. Expect headers like 'Pelvis q0..q3'.")

    # Convert to Euler
    eul_df = build_euler_dataframe(df, joint_cols, frame_col=args.frame_col)

    # Plots
    plot_time_series(eul_df, args.key_joints, frame_col=args.frame_col, outdir=outdir)
    plot_correlation(eul_df, outdir=outdir)
    plot_pca(eul_df, outdir=outdir)
    if args.fft_signal:
        plot_periodicity(eul_df, args.fft_signal, outdir=outdir)

    # Quick textual hints
    # (You can expand this with change-point detection or clustering if needed.)
    print(f"Done. Wrote figures to: {outdir.resolve()}")
    print("Tips:")
    print("- Repeated peaks in FFT plots usually indicate cyclic motion (e.g., gait).")
    print("- Strong off-diagonal blocks in the correlation heatmap suggest joints moving together.")
    print("- Clear structure in PC1/PC2 scatter means low-dimensional patterns are present.")

if __name__ == "__main__":
    main()
