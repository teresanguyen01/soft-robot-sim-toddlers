import glob 
import os
import pandas as pd
import numpy as np
import warnings
import argparse

def report_nans(mocap_dir): 
    files = sorted(f for f in glob.glob(os.path.join(mocap_dir, '*.csv')))

    print(f"Checking {len(files)} mocap files for missing values...\n")

    for f in files: 
        df = pd.read_csv(f)
        total_nans = int(df.isna().sum().sum())
        if total_nans == 0: 
            print(f"{os.path.basename(f):30s} is OK: no missing data")
        else: 
            col_nans = df.isna().sum()
            print(f"{os.path.basename(f):30s} {total_nans} total Nans")
            for col, count in col_nans[col_nans > 0].items(): 
                print(f"{col:20s} -> {count}")
    print("Finished checking for NaNs")

def load_data(sensor_dir, mocap_dir): 
    sensor_files = sorted(f for f in glob.glob(os.path.join(sensor_dir, '*sensor*.csv')))
    X_dfs, y_dfs = [], []

    for sf in sensor_files: 
        bn = os.path.basename(sf)
        base = bn.replace('_sensor', '').rsplit('.csv', 1)[0]
        pattern = os.path.join(mocap_dir, f"{base}*resamp*.csv")
        matches = glob.glob(pattern)
        mf = matches[0] if matches else os.path.join(mocap_dir, bn.replace('_sensor', ''))

        if not os.path.exists(mf): 
            print(f"MISSING MOCAP FOR {bn}, SKIPPING")
            continue
        
        df_X = pd.read_csv(sf)
        df_y = pd.read_csv(mf)
        print(f"\nLoaded pair: {bn} <-> {os.path.basename(mf)}")
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

    common_mocap_cols = [c for c in common_mocap_cols
                         if c not in ('time_ms','time_s','Time_ms')]

    drop = {'base_X', 'base_Y', 'base_Z','base_W','base_X.1', 'base_Y.1', 'base_Z.1'}
    common_mocap_cols = [c for c in common_mocap_cols if c not in drop]
    common_sensor_cols = [c for c in common_sensor_cols if c not in ('time_ms','time_s', 'Time_ms')]


    print(f"Keeping {len(common_sensor_cols)} common sensor cols and {len(common_mocap_cols)} common mocap cols")

    X_all = pd.concat([df[common_sensor_cols] for df in X_dfs], axis=0, ignore_index=True)
    y_all = pd.concat([df[common_mocap_cols] for df in y_dfs], axis=0, ignore_index=True)

    return X_all.values, y_all.values, common_sensor_cols, common_mocap_cols

def main(): 
    p = argparse.ArgumentParser(description='Train sensor to mocap regression model')
    p.add_argument('--sensor_dir', required=True, help='Directory of sensor files')
    p.add_argument('--mocap_dir', required=True, help='Directory of mocap files')
    p.add_argument('--output_dir', required=True, help='Directory to save model and metrics')
    p.add_argument('--test_size', type=float, default=0.2, help='amount of data for validation')
    p.add_argument('--random_state', type=int, default=42, help='random seed')

    args = p.parse_args()
    report_nans(args.mocap_dir)
    X, y, sensor_cols, mocap_cols = load_data(args.sensor_dir, args.mocap_dir)

    X_all, y_all, sensor_cols, mocap_cols = load_data(args.sensor_dir, args.mocap_dir)

    df_X_all = pd.DataFrame(X_all, columns=sensor_cols)
    df_y_all = pd.DataFrame(y_all, columns=mocap_cols)
    sensor_path = os.path.join(args.output_dir, 'sensor_inputs.csv')
    df_X_all.to_csv(sensor_path, index=False)
    print(f"Saved sensor inputs to {sensor_path}")

    # save mocap targets alone
    mocap_path = os.path.join(args.output_dir, 'mocap_targets.csv')
    df_y_all.to_csv(mocap_path, index=False)
    print(f"Saved mocap targets to {mocap_path}")

if __name__ == "__main__":
    main()
