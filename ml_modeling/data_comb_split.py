#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Tuple, Optional, List

import pandas as pd

TIME_KEYS = ["time", "timestamp", "Time", "Timestamp", "time_ms", "Frame", "frame"]

def list_csvs_by_prefix(directory: str) -> Dict[str, str]:
    mapping = {}
    for fn in os.listdir(directory):
        if not fn.lower().endswith(".csv"):
            continue
        prefix = fn.split("_", 1)[0]
        mapping[prefix] = os.path.join(directory, fn)
    return mapping

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def find_common_time_key(df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[str]:
    df1_keys = set(df1.columns)
    df2_keys = set(df2.columns)
    for key in TIME_KEYS:
        if key in df1_keys and key in df2_keys:
            return key
    return None

def align_dataframes(sensor_df: pd.DataFrame, mocap_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = find_common_time_key(sensor_df, mocap_df)
    if key is not None:
        merged = pd.merge(sensor_df, mocap_df, on=key, how="inner", suffixes=("_sensor", "_mocap"))
        sensor_cols = [c for c in sensor_df.columns if c != 'time_ms']+  [key]
        mocap_cols  = [c for c in mocap_df.columns if c != 'time_ms']  + [key]
        sensor_df_aligned = merged[sensor_cols].rename(columns=lambda c: c[:-7] if c.endswith("_sensor") else c)
        mocap_df_aligned  = merged[mocap_cols ].rename(columns=lambda c: c[:-6] if c.endswith("_mocap")  else c)
        try:
            sensor_df_aligned = sensor_df_aligned.sort_values(by=key).reset_index(drop=True)
            mocap_df_aligned  = mocap_df_aligned.sort_values(by=key).reset_index(drop=True)
        except Exception:
            sensor_df_aligned = sensor_df_aligned.reset_index(drop=True)
            mocap_df_aligned  = mocap_df_aligned.reset_index(drop=True)
        return sensor_df_aligned, mocap_df_aligned
    else:
        n = min(len(sensor_df), len(mocap_df))
        if n == 0:
            raise ValueError("One of the paired files has zero rows after reading.")
        return sensor_df.iloc[:n].reset_index(drop=True), mocap_df.iloc[:n].reset_index(drop=True)

def chronological_split(df_sensor: pd.DataFrame, df_mocap: pd.DataFrame, train_ratio: float = 0.7):
    assert len(df_sensor) == len(df_mocap), "Aligned dataframes must have equal length"
    n = len(df_sensor)
    split_idx = max(1, min(n - 1, int(n * train_ratio)))
    sensor_train = df_sensor.iloc[:split_idx].reset_index(drop=True)
    sensor_test  = df_sensor.iloc[split_idx:].reset_index(drop=True)
    mocap_train  = df_mocap.iloc[:split_idx].reset_index(drop=True)
    mocap_test   = df_mocap.iloc[split_idx:].reset_index(drop=True)
    return sensor_train, mocap_train, sensor_test, mocap_test

def select_numeric_only(df: pd.DataFrame, preserve_keys: Optional[List[str]] = None) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    print("DId it run", numeric)
    if preserve_keys:
        extras = [k for k in preserve_keys if k in df.columns and k not in numeric.columns]
        return pd.concat([numeric, df[extras]], axis=1)
    return numeric

def main():
    parser = argparse.ArgumentParser(description="Pair sensor & mocap CSVs, align, and 70/30 split.")
    parser.add_argument("--sensor_dir", required=True)
    parser.add_argument("--mocap_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    # NEW: only drop to numeric if you ask for it
    parser.add_argument("--numeric_only", action="store_true",
                        help="If set, keep numeric columns only (plus any shared time key).")
    args = parser.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    sensor_map = list_csvs_by_prefix(args.sensor_dir)
    mocap_map  = list_csvs_by_prefix(args.mocap_dir)

    common_prefixes = sorted(set(sensor_map) & set(mocap_map))
    if not common_prefixes:
        print("No matching prefixes found between sensor and mocap directories.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(common_prefixes)} matching pairs: {common_prefixes}")

    for prefix in common_prefixes:
        sensor_path = sensor_map[prefix]
        mocap_path  = mocap_map[prefix]
        print(f"\nProcessing pair: {prefix}\n  Sensor: {sensor_path}\n  Mocap : {mocap_path}")

        sensor_df = read_csv_safe(sensor_path)
        mocap_df  = read_csv_safe(mocap_path)

        sensor_aligned, mocap_aligned = align_dataframes(sensor_df, mocap_df)

        if args.numeric_only:
            key = find_common_time_key(sensor_aligned, mocap_aligned)
            sensor_aligned = select_numeric_only(sensor_aligned, [key] if key else None)
            mocap_aligned  = select_numeric_only(mocap_aligned,  [key] if key else None)
        # else: KEEP ALL COLUMNS

        s_train, m_train, s_test, m_test = chronological_split(
            sensor_aligned, mocap_aligned, train_ratio=args.train_ratio
        )

        train_sensor_fn = os.path.join(args.train_dir, f"{prefix}_training_sensor.csv")
        train_mocap_fn  = os.path.join(args.train_dir, f"{prefix}_training_mocap.csv")
        test_sensor_fn  = os.path.join(args.test_dir,  f"{prefix}_testing_sensor.csv")
        test_mocap_fn   = os.path.join(args.test_dir,  f"{prefix}_testing_mocap.csv")

        s_train.to_csv(train_sensor_fn, index=False)
        m_train.to_csv(train_mocap_fn,  index=False)
        s_test.to_csv(test_sensor_fn,   index=False)
        m_test.to_csv(test_mocap_fn,    index=False)

        print("Saved:")
        print(f"  {train_sensor_fn}")
        print(f"  {train_mocap_fn}")
        print(f"  {test_sensor_fn}")
        print(f"  {test_mocap_fn}")

    print("\nDone.")

if __name__ == "__main__":
    main()
