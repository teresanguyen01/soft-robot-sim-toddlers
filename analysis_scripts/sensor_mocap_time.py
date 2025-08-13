#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ALLOWED_EXT = (".csv", ".tsv", ".txt")

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT with delimiter by extension."""
    suf = path.suffix.lower()
    sep = "," if suf == ".csv" else "\t"
    try:
        return pd.read_csv(path, sep=sep)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, engine="python")

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def find_angle_match(angle_dir: Path, base_stem: str) -> Path | None:
    """Find angle file in angle_dir whose stem equals base_stem (without _sensor)."""
    candidates = [p for p in angle_dir.iterdir()
                  if p.is_file()
                  and p.suffix.lower() in ALLOWED_EXT
                  and p.stem == base_stem]
    if not candidates:
        low = base_stem.lower()
        candidates = [p for p in angle_dir.iterdir()
                      if p.is_file()
                      and p.suffix.lower() in ALLOWED_EXT
                      and p.stem.lower() == low]
    if not candidates:
        return None
    csvs = [p for p in candidates if p.suffix.lower() == ".csv"]
    return (csvs[0] if csvs else candidates[0])

def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """
    For each t in target_times, return index i such that source_times[i] is closest to t.
    Assumes source_times is sorted ascending and finite.
    """
    pos = np.searchsorted(source_times, target_times)
    pos_right = np.clip(pos, 0, len(source_times) - 1)
    pos_left = np.clip(pos - 1, 0, len(source_times) - 1)
    left_diff = np.abs(target_times - source_times[pos_left])
    right_diff = np.abs(source_times[pos_right] - target_times)
    choose_right = right_diff < left_diff
    return np.where(choose_right, pos_right, pos_left)

def align_one_pair(angle_path: Path, sensor_path: Path, keep_angle_time=False) -> pd.DataFrame:
    """
    Load angle & sensor, align angle rows to nearest sensor time_ms. Return aligned angle df
    with length == len(sensor), time_ms set to sensor time_ms.
    """
    ang = read_table(angle_path)
    sen = read_table(sensor_path)

    if "time_ms" not in ang.columns:
        raise ValueError(f"{angle_path.name} missing 'time_ms'")
    if "time_ms" not in sen.columns:
        raise ValueError(f"{sensor_path.name} missing 'time_ms'")

    ang = ang.sort_values("time_ms").reset_index(drop=True)
    sen = sen.sort_values("time_ms").reset_index(drop=True)

    ang_time = pd.to_numeric(ang["time_ms"], errors="coerce").to_numpy(dtype=float)
    sen_time = pd.to_numeric(sen["time_ms"], errors="coerce").to_numpy(dtype=float)

    if np.isnan(ang_time).all():
        raise ValueError(f"All 'time_ms' are NaN in {angle_path.name}")
    if np.isnan(sen_time).all():
        raise ValueError(f"All 'time_ms' are NaN in {sensor_path.name}")

    ang_valid = ~np.isnan(ang_time)
    ang = ang.loc[ang_valid].reset_index(drop=True)
    ang_time = ang_time[ang_valid]

    sen_valid = ~np.isnan(sen_time)
    sen = sen.loc[sen_valid].reset_index(drop=True)
    sen_time = sen_time[sen_valid]

    idx = nearest_indices(ang_time, sen_time)

    angle_cols = [c for c in ang.columns if c != "time_ms"]
    aligned = ang.loc[idx, angle_cols].reset_index(drop=True)
    aligned.insert(0, "time_ms", sen_time)

    if keep_angle_time:
        aligned.insert(1, "angle_time_ms_chosen", ang_time[idx])

    return aligned

def main():
    ap = argparse.ArgumentParser(
        description="Align angle arrays to nearest sensor timestamps using time_ms (one output per matched pair)."
    )
    ap.add_argument("--angle_dir", help="Folder containing angle array files (e.g., Emma.csv)")
    ap.add_argument("--sensor_dir", help="Folder containing sensor files (same name but with '_sensor' suffix)")
    ap.add_argument("--output_dir", help="Folder to write aligned angle arrays")
    ap.add_argument("--keep-angle-time", action="store_true",
                    help="Add 'angle_time_ms_chosen' column for diagnostics")
    args = ap.parse_args()

    angle_dir = Path(args.angle_dir)
    sensor_dir = Path(args.sensor_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not angle_dir.is_dir():
        sys.exit(f"Angle dir not found: {angle_dir}")
    if not sensor_dir.is_dir():
        sys.exit(f"Sensor dir not found: {sensor_dir}")

    sensor_files = [p for p in sensor_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in ALLOWED_EXT
                    and re.search(r"(?:^|_)sensor$", p.stem, flags=re.IGNORECASE)]

    if not sensor_files:
        print("No sensor files matching '*_sensor.(csv|tsv|txt)' found.", file=sys.stderr)
        sys.exit(1)

    processed = 0
    skipped = 0

    for s_path in sorted(sensor_files):
        base_stem = re.sub(r"(?:^|_)(sensor)$", "", s_path.stem, count=1, flags=re.IGNORECASE)
        if base_stem.endswith("_"):
            base_stem = base_stem[:-1]

        a_path = find_angle_match(angle_dir, base_stem)
        if a_path is None:
            print(f"[skip] No angle file for sensor '{s_path.name}' (expected stem '{base_stem}')", file=sys.stderr)
            skipped += 1
            continue

        try:
            aligned = align_one_pair(a_path, s_path, keep_angle_time=args.keep_angle_time)
        except Exception as e:
            print(f"[error] {s_path.name} x {a_path.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        out_name = f"{base_stem}_angles_aligned_to_sensor.csv"
        out_path = out_dir / out_name
        write_csv(aligned, out_path)
        print(f"[ok] Wrote {out_path} (rows: {len(aligned)})")
        processed += 1

    if processed == 0:
        sys.exit("No pairs processed. Check file names and presence of 'time_ms' columns.")
    else:
        print(f"Done. Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
