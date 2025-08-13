import glob
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def report_nans(mocap_dir):
    files = sorted(glob.glob(os.path.join(mocap_dir, '*.csv')))
    print(f"Checking {len(files)} mocap files for missing values...\n")

    for f in files:
        df = pd.read_csv(f)
        total_nans = int(df.isna().sum().sum())
        if total_nans == 0:
            print(f"{os.path.basename(f):30s} is OK: no missing data")
        else:
            col_nans = df.isna().sum()
            print(f"{os.path.basename(f):30s} {total_nans} total NaNs")
            for col, count in col_nans[col_nans > 0].items():
                print(f"{col:20s} -> {count}")
    print("Finished checking for NaNs")

def _person_from_sensor_filename(sensor_basename: str) -> str:
    """Given 'alice_sensor.csv' -> 'alice'"""
    stem, _ = os.path.splitext(sensor_basename)
    if stem.endswith('_sensor'):
        return stem[:-8]  # strip '_sensor'
    return stem

def load_data(sensor_dir, mocap_dir):
    allowed_names = {"adrian", "muhammad", "emma"}

    sensor_files = []
    for name in allowed_names:
        sensor_files.extend(glob.glob(os.path.join(sensor_dir, f"{name}_sensor.csv")))
    sensor_files = sorted(sensor_files)

    X_dfs, y_dfs = [], []

    for sf in sensor_files:
        sensor_bn = os.path.basename(sf)
        person = sensor_bn.replace("_sensor.csv", "")

        mocap_path = os.path.join(mocap_dir, f"{person}.csv")
        if not os.path.exists(mocap_path):
            print(f"MISSING MOCAP FOR {sensor_bn}, expected '{person}.csv', SKIPPING")
            continue

        df_X = pd.read_csv(sf)
        df_y = pd.read_csv(mocap_path)
        print(f"\nLoaded pair: {sensor_bn} <-> {os.path.basename(mocap_path)}")
        print(f"rows: sensor={len(df_X)}, mocap={len(df_y)}")

        if len(df_X) != len(df_y):
            print("Row count mismatch, skipping this pair")
            continue

        X_dfs.append(df_X)
        y_dfs.append(df_y)

    if not X_dfs:
        raise RuntimeError("No valid sensor/mocap pairs found")

    common_sensor_cols = sorted(set.intersection(*(set(df.columns) for df in X_dfs)))
    common_mocap_cols = sorted(set.intersection(*(set(df.columns) for df in y_dfs)))

    time_cols = {'time_ms', 'time_s', 'Time_ms'}
    drop = {'base_X', 'base_Y', 'base_Z', 'base_W', 'base_X.1', 'base_Y.1', 'base_Z.1'}

    common_sensor_cols = [c for c in common_sensor_cols if c not in time_cols]
    common_mocap_cols = [c for c in common_mocap_cols if c not in time_cols and c not in drop]

    print(f"Keeping {len(common_sensor_cols)} common sensor cols and {len(common_mocap_cols)} common mocap cols")

    X_all = pd.concat([df[common_sensor_cols] for df in X_dfs], axis=0, ignore_index=True)
    y_all = pd.concat([df[common_mocap_cols] for df in y_dfs], axis=0, ignore_index=True)

    return X_all.values, y_all.values, common_sensor_cols, common_mocap_cols

def main():
    p = argparse.ArgumentParser(description='Prepare aligned sensor/mocap datasets with train/test split')
    p.add_argument('--sensor_dir', required=True, help='Directory of sensor files (e.g., alice_sensor.csv)')
    p.add_argument('--mocap_dir', required=True, help='Directory of mocap files (e.g., alice.csv)')
    p.add_argument('--output_dir', required=True, help='Directory to save merged inputs/targets')
    p.add_argument('--test_size', type=float, default=0.3, help='Fraction of data for testing')
    p.add_argument('--random_state', type=int, default=42, help='Random seed')

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    report_nans(args.mocap_dir)
    X_all, y_all, sensor_cols, mocap_cols = load_data(args.sensor_dir, args.mocap_dir)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )

    # Save train files
    pd.DataFrame(X_train, columns=sensor_cols).to_csv(os.path.join(args.output_dir, 'sensor_train.csv'), index=False)
    pd.DataFrame(y_train, columns=mocap_cols).to_csv(os.path.join(args.output_dir, 'mocap_train.csv'), index=False)

    # Save test files
    pd.DataFrame(X_test, columns=sensor_cols).to_csv(os.path.join(args.output_dir, 'sensor_test.csv'), index=False)
    pd.DataFrame(y_test, columns=mocap_cols).to_csv(os.path.join(args.output_dir, 'mocap_test.csv'), index=False)

    print("✅ Saved sensor_train.csv, mocap_train.csv, sensor_test.csv, mocap_test.csv")

if __name__ == "__main__":
    main()
