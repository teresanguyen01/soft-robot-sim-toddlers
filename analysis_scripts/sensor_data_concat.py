#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Filenames like "4_Muhammad_Whatever.csv"
# Group 1 = number, Group 2 = name
FILE_RE = re.compile(r"^(\d+)_([A-Za-z]+)_.+\.(csv|tsv|txt)$", re.IGNORECASE)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Concatenate given input files by name (e.g., Emma/Adrian/Muhammad) in numeric order (1_, 2_, ...), "
            "then replace timestamps with a uniform time_ms sequence."
        )
    )
    p.add_argument(
        "inputs", nargs="+",
        help="Input files (e.g., 1_Emma_*.csv 2_Emma_*.csv 1_Adrian_*.csv ...)"
    )
    p.add_argument(
        "-o", "--output_dir", default=".",
        help="Directory to write <Name>_concatenated_clean.csv files (default: current dir)"
    )
    p.add_argument(
        "--outlier-factor", type=float, default=1.5,
        help="IQR multiplier for numeric-column outlier masking (default: 1.5)"
    )
    p.add_argument(
        "--no-interp", action="store_true",
        help="Do NOT interpolate after outlier removal"
    )
    p.add_argument(
        "--drop-debug", action="store_true",
        help="Drop __segment_id__ and __source__ columns in the final CSV"
    )
    p.add_argument(
        "--keep-original-timestamp", action="store_true",
        help="Keep original timestamp column as 'Timestamp_original' in the output"
    )
    return p.parse_args()

# ------------------------------------------------------------
# IO / Utility
# ------------------------------------------------------------
def parse_name_and_order(path: Path) -> Optional[Tuple[str, int]]:
    m = FILE_RE.match(path.name)
    if not m:
        return None
    order = int(m.group(1))
    name = m.group(2)
    return name, order

def read_table(path: Path) -> pd.DataFrame:
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    try:
        df = pd.read_csv(path, sep=sep)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, engine="python")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Drop auto index columns if present
    drop_these = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if drop_these:
        df = df.drop(columns=drop_these)
    return df

def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    # Look for a likely timestamp column (case-insensitive)
    candidates = ["timestamp", "time", "time_ms"]
    lower_map = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return None

# ------------------------------------------------------------
# Time handling
# ------------------------------------------------------------
def parse_hhmmssms_to_ms(series: pd.Series) -> pd.Series:
    """Parse 'HH:MM:SS:ms' -> milliseconds since midnight (float, NaNs preserved)."""
    def parse_one(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        parts = s.split(":")
        if len(parts) != 4:
            return np.nan
        try:
            hh, mm, ss, ms = map(int, parts)
        except Exception:
            return np.nan
        return float((hh*3600 + mm*60 + ss) * 1000 + ms)
    return series.map(parse_one).astype("float64")

def to_ms_vector(ts: pd.Series) -> pd.Series:
    """
    Accepts either HH:MM:SS:ms strings OR already-numeric milliseconds.
    Returns float64 ms with NaNs preserved.
    """
    if pd.api.types.is_numeric_dtype(ts):
        return ts.astype("float64")
    # Try to parse HH:MM:SS:ms
    return parse_hhmmssms_to_ms(ts)

def fix_rollover_within_segments(ms_vals: np.ndarray, seg_ids: np.ndarray) -> np.ndarray:
    """
    For each segment (file), fix midnight rollovers: when time decreases, add +24h.
    Operates on a copy and returns it.
    """
    out = ms_vals.copy()
    day_ms = 86_400_000.0
    for seg in pd.unique(seg_ids):
        idx = np.where(seg_ids == seg)[0]
        if len(idx) < 2:
            continue
        add = 0.0
        for j in range(1, len(idx)):
            i_prev, i_cur = idx[j-1], idx[j]
            a, b = out[i_prev], out[i_cur]
            if np.isnan(a) or np.isnan(b):
                continue
            if b < a:  # rollover within this file
                add += day_ms
            out[i_cur] += add
    return out

def build_uniform_time_from_timestamp(ts: pd.Series, segment_ids: Optional[pd.Series]) -> pd.Series:
    """
    AFTER concatenation:
      1) parse timestamp (HH:MM:SS:ms or numeric ms) to ms
      2) correct midnight rollovers within each segment
      3) estimate constant step using positive adjacent diffs within segments
      4) return uniform vector: 0, step, 2*step, ...
    """
    base = to_ms_vector(ts)
    if base.notna().sum() == 0:
        return pd.Series(np.nan, index=ts.index, dtype="float64")

    seg_ids = segment_ids if segment_ids is not None else pd.Series(0, index=ts.index)
    seg_ids = seg_ids.to_numpy()
    vals = fix_rollover_within_segments(base.to_numpy(), seg_ids)

    # Collect positive adjacent diffs WITHIN segments
    diffs: List[float] = []
    for seg in np.unique(seg_ids):
        idx = np.where(seg_ids == seg)[0]
        if len(idx) < 2:
            continue
        v = vals[idx]
        for k in range(1, len(v)):
            a, b = v[k-1], v[k]
            if not np.isnan(a) and not np.isnan(b):
                d = b - a
                if d > 0:
                    diffs.append(d)

    if len(diffs) == 0:
        step = 0.0
    else:
        step = float(np.median(diffs))  # robust across segments
        if step <= 0:
            step = float(diffs[0])

    n = len(vals)
    uniform = np.arange(n, dtype="float64") * step
    return pd.Series(np.rint(uniform).astype("int64"), index=ts.index)

# ------------------------------------------------------------
# Outlier handling
# ------------------------------------------------------------
def remove_outliers_per_column(df: pd.DataFrame, exclude=(), factor: float = 1.5, interpolate=True) -> pd.DataFrame:
    """
    IQR filter per numeric column; outliers -> NaN; optional linear interpolation.
    """
    out = df.copy()
    num_cols = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        mask = ~(out[c].between(lo, hi) | out[c].isna())
        out.loc[mask, c] = np.nan
        if interpolate:
            out[c] = out[c].interpolate(method="linear", limit_direction="both")
    return out

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for s in args.inputs:
        p = Path(s)
        if not p.is_file():
            raise SystemExit(f"Input is not a file or doesn't exist: {p}")
        parsed = parse_name_and_order(p)
        if parsed is None:
            raise SystemExit(f"Filename does not match 'N_Name_*.csv/tsv/txt' pattern: {p.name}")
        name, order = parsed
        groups.setdefault(name, []).append((order, p))

    if not groups:
        raise SystemExit("No valid inputs matched the expected filename pattern.")

    for name in sorted(groups.keys()):
        parts = sorted(groups[name], key=lambda t: (t[0], t[1].name.lower()))
        print(f"\n==> {name}: {len(parts)} file(s)")
        for order, path in parts:
            print(f"   - {order:>3} :: {path.name}")

        # 1) Read all files (no timestamp rewriting yet)
        dfs: List[pd.DataFrame] = []
        for order, path in parts:
            df = read_table(path)
            ts_col = find_timestamp_column(df)
            if ts_col is None:
                raise SystemExit(f"{path.name} has no timestamp-like column (looked for: Timestamp/Time/time_ms).")
            # Normalize to a single 'Timestamp' column name
            if ts_col != "Timestamp":
                df = df.rename(columns={ts_col: "Timestamp"})
            # Debug columns
            df["__segment_id__"] = order
            df["__source__"] = path.name
            dfs.append(df)

        # 2) UNION of columns (keep everything; fill missing with NaN)
        col_order: List[str] = []
        seen = set()
        for d in dfs:
            print(d)
            for c in d.columns:
                if c not in seen:
                    seen.add(c)
                    col_order.append(c)
        dfs = [d.reindex(columns=col_order) for d in dfs]

        # 3) CONCATENATE in numeric order (1_, 2_, 3_, ...)
        cat = pd.concat(dfs, ignore_index=True, sort=False)

        # 4) AFTER concatenation, build uniform time from the concatenated Timestamp
        if args.keep_original_timestamp:
            cat.rename(columns={"Timestamp": "Timestamp_original"}, inplace=True)
            ts_col = "Timestamp_original"
        else:
            ts_col = "Timestamp"

        cat["time_ms"] = build_uniform_time_from_timestamp(cat[ts_col], segment_ids=cat["__segment_id__"])

        # # 5) Optional outlier removal (does NOT touch time_ms or debug cols, or timestamp_original)
        # exclude_cols = ["time_ms", "__segment_id__", "__source__"]
        # if args.keep_original_timestamp:
        #     exclude_cols.append("Timestamp_original")
        # clean = remove_outliers_per_column(
        #     cat, exclude=tuple(exclude_cols),
        #     factor=args.outlier_factor, interpolate=not args.no_interp
        # )

        # # 6) Optionally drop debug meta columns
        # if args.drop_debug:
        cat = cat.drop(columns=["__segment_id__", "__source__", "Timestamp"])

        out_path = out_dir / f"{name}_concatenated_clean.csv"
        cat.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  (rows={len(cat)}, cols={len(cat.columns)})")

    print("\nDone.")

if __name__ == "__main__":
    main()
