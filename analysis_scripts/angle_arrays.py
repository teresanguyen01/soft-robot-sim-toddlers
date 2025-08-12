#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

# === 1) DEFINE WHICH COLUMNS YOU WANT AND HOW TO RENAME THEM ===
# Map raw column -> target column name (unique)
RENAME_MAPPING: Dict[str, str] = {
    # Right shoulder
    'Right T4 Shoulder Abduction/Adduction': 'right_shoulder_roll_joint',
    'Right T4 Shoulder Internal/External Rotation': 'right_shoulder_yaw_joint',
    'Right T4 Shoulder Flexion/Extension': 'right_shoulder_pitch_joint',
    'Right Shoulder Abduction/Adduction': 'right_shoulder_roll_joint',
    'Right Shoulder Internal/External Rotation': 'right_shoulder_yaw_joint',
    'Right Shoulder Flexion/Extension': 'right_shoulder_pitch_joint',
    'Right Elbow Flexion/Extension': 'right_elbow_pitch_joint',
    # Left shoulder
    'Left T4 Shoulder Abduction/Adduction': 'left_shoulder_roll_joint',
    'Left T4 Shoulder Internal/External Rotation': 'left_shoulder_yaw_joint',
    'Left T4 Shoulder Flexion/Extension': 'left_shoulder_pitch_joint',
    'Left Shoulder Abduction/Adduction': 'left_shoulder_roll_joint',
    'Left Shoulder Internal/External Rotation': 'left_shoulder_yaw_joint',
    'Left Shoulder Flexion/Extension': 'left_shoulder_pitch_joint',
    'Left Elbow Flexion/Extension': 'left_elbow_pitch_joint',
    # Hips
    'Right Hip Abduction/Adduction': 'right_hip_roll_joint',
    'Right Hip Internal/External Rotation': 'right_hip_yaw_joint',
    'Right Hip Flexion/Extension': 'right_hip_pitch_joint',
    'Left Hip Abduction/Adduction': 'left_hip_roll_joint',
    'Left Hip Internal/External Rotation': 'left_hip_yaw_joint',
    'Left Hip Flexion/Extension': 'left_hip_pitch_joint',
}

TARGETS: List[str] = list(dict.fromkeys(RENAME_MAPPING.values()))  # unique, keep order


def read_csv_basic(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def keep_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns present in RENAME_MAPPING (plus Frame/time_ms if present),
    then rename to target names. If multiple sources map to the same target,
    collapse by taking the first non-null across duplicates.
    """
    # Keep only mapped sources + time helpers
    keep_cols = [c for c in RENAME_MAPPING.keys() if c in df.columns]
    time_helpers = [c for c in ['Frame', 'time_ms'] if c in df.columns]
    df = df[keep_cols + time_helpers].copy()

    # Rename mapped columns
    df = df.rename(columns=RENAME_MAPPING)

    # Collapse duplicates to ensure unique target columns
    out = pd.DataFrame(index=df.index)
    for tgt in TARGETS:
        same = [c for c in df.columns if c == tgt]
        if not same:
            continue
        if len(same) == 1:
            out[tgt] = pd.to_numeric(df[same[0]], errors='coerce')
        else:
            # first non-null across duplicates (left-to-right)
            sub = df[same].apply(pd.to_numeric, errors='coerce')
            out[tgt] = sub.bfill(axis=1).iloc[:, 0]

    # Append time helper columns (not converted to radians)
    for c in time_helpers:
        out[c] = pd.to_numeric(df[c], errors='coerce')

    return out


def convert_targets_to_radians(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ONLY the target joint columns to radians (leave time columns alone)."""
    out = df.copy()
    for tgt in TARGETS:
        if tgt in out.columns:
            out[tgt] = np.deg2rad(pd.to_numeric(out[tgt], errors='coerce'))
    return out


def ensure_time_ms(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    If time_ms exists, keep it.
    Else if Frame exists, compute time_ms = Frame * (1000/fps).
    Else create a uniform time_ms assuming 1/fps sampling.
    """
    out = df.copy()
    if 'time_ms' in out.columns and out['time_ms'].notna().any():
        # already present — do nothing
        return out
    if 'Frame' in out.columns and out['Frame'].notna().any():
        out['time_ms'] = pd.to_numeric(out['Frame'], errors='coerce') * (1000.0 / fps)
        out = out.drop(columns=['Frame'])
        return out
    # Fallback: uniform timeline
    out['time_ms'] = np.arange(len(out), dtype=float) * (1000.0 / fps)
    return out


def process_file(in_path: str, out_path: str, fps: float) -> None:
    df = read_csv_basic(in_path)
    df = keep_and_rename(df)              # drop others, rename + collapse dupes
    df = convert_targets_to_radians(df)   # degrees -> radians
    df = ensure_time_ms(df, fps=fps)      # compute time_ms
    # Reorder columns: time_ms first, then targets (only those present)
    cols = ['time_ms'] + [c for c in TARGETS if c in df.columns]
    df = df[cols]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Drop unused cols, rename, convert to radians, compute time_ms.")
    ap.add_argument('--input_dir', required=True, help='Folder with input CSV files')
    ap.add_argument('--output_dir', required=True, help='Folder to save processed CSV files')
    ap.add_argument('--fps', type=float, default=60.0, help='Frames per second for Frame -> time_ms (default: 60)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for name in os.listdir(args.input_dir):
        if not name.lower().endswith('.csv'):
            continue
        in_path = os.path.join(args.input_dir, name)
        out_path = os.path.join(args.output_dir, name)
        process_file(in_path, out_path, fps=args.fps)
        print(f"[ok] {name}")

if __name__ == '__main__':
    main()
