#!/usr/bin/env python3
"""
Drop rows with values past a threshold.

Rule (default):
  If ANY numeric column in a row is > 500, drop that row.

Options:
  --threshold 700         # change the cutoff (default: 500)
  --absolute              # use |value| > threshold instead of value > threshold
  --ignore Timestamp time time_ms   # columns never checked
  --pattern *.csv *.tsv   # which files to process
  --recursive             # search subfolders
  --outdir ./clean        # where to save cleaned files
  --suffix _clean         # filename suffix (default: _clean)
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove rows where any numeric value exceeds a threshold.")
    p.add_argument("--input_dir", required=True, help="Directory containing input files.")
    p.add_argument("--pattern", nargs="*", default=["*.csv"], help="Glob patterns (default: *.csv).")
    p.add_argument("--recursive", action="store_true", help="Search recursively.")
    p.add_argument("--output_dir", default=None, help="Output directory (default: alongside inputs).")
    p.add_argument("--suffix", default="_clean", help="Suffix for cleaned files (default: _clean).")
    p.add_argument("--threshold", type=float, default=500.0, help="Threshold (default: 500).")
    p.add_argument("--absolute", action="store_true",
                   help="Use absolute value: drop if |value| > threshold. Default is value > threshold.")
    p.add_argument("--ignore", nargs="*", default=["Timestamp", "time", "time_ms"],
                   help="Column names to ignore (case-sensitive).")
    return p.parse_args()


def find_files(input_dir: Path, patterns: List[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(input_dir.rglob(pat) if recursive else input_dir.glob(pat))
    # dedupe, keep stable order, only files
    out, seen = [], set()
    for f in files:
        p = f.resolve()
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def read_table(path: Path) -> pd.DataFrame:
    # Try delimiter sniffing; fall back to comma
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    drop_auto = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if drop_auto:
        df = df.drop(columns=drop_auto)
    return df


def make_output_path(in_path: Path, outdir: Path | None, suffix: str) -> Path:
    odir = outdir if outdir is not None else in_path.parent
    odir.mkdir(parents=True, exist_ok=True)
    return odir / f"{in_path.stem}{suffix}{in_path.suffix}"


def drop_rows_over_threshold(df: pd.DataFrame, threshold: float, absolute: bool, ignore_cols: List[str]) -> tuple[pd.DataFrame, int]:
    # Coerce a numeric view for checking only (preserve original df values)
    check_cols = [c for c in df.columns if c not in ignore_cols]
    num_view = {}
    for c in check_cols:
        s = df[c]
        # only consider as numeric if we can coerce at least some numbers
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            num_view[c] = s_num
    if not num_view:
        return df.copy(), 0

    num_df = pd.DataFrame(num_view, index=df.index)

    if absolute:
        mask = (num_df.abs() > threshold).any(axis=1)
    else:
        mask = (num_df > threshold).any(axis=1)

    cleaned = df.loc[~mask].copy()
    return cleaned, int(mask.sum())


def main():
    args = parse_args()
    indir = Path(args.input_dir).resolve()
    if not indir.is_dir():
        print(f"ERROR: not a directory: {indir}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.output_dir).resolve() if args.output_dir else None
    files = find_files(indir, args.pattern, args.recursive)
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files. Threshold={args.threshold} "
          f"({'abs' if args.absolute else '>'}) Ignore={args.ignore or '[]'}")

    total_in = total_out = total_drop = 0

    for path in files:
        try:
            df = read_table(path)
        except Exception as e:
            print(f"[SKIP] {path.name}: read error: {e}", file=sys.stderr)
            continue

        rows_in = len(df)
        cleaned, dropped = drop_rows_over_threshold(
            df, threshold=args.threshold, absolute=args.absolute, ignore_cols=args.ignore
        )
        rows_out = len(cleaned)
        out_path = make_output_path(path, outdir, args.suffix)
        try:
            cleaned.to_csv(out_path, index=False)
        except Exception as e:
            print(f"[ERROR] write failed for {out_path}: {e}", file=sys.stderr)
            continue

        total_in += rows_in
        total_out += rows_out
        total_drop += dropped
        print(f"[OK] {path.name}  rows_in={rows_in}  dropped={dropped}  rows_out={rows_out}  -> {out_path.name}")

    print(f"\nDONE. files={len(files)}  total_in={total_in}  dropped={total_drop}  total_out={total_out}")


if __name__ == "__main__":
    main()
