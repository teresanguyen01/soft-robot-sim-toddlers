import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _remove_outliers_mad(series, mad_z=3.5, fallback_pct=(0.5, 99.5)):
    """
    Remove outliers from a 1D pandas Series using MAD.
    Values with |z_mad| > mad_z are set to NaN.
    If MAD is 0/NaN, fall back to percentile trimming.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        lo = np.nanpercentile(x, fallback_pct[0])
        hi = np.nanpercentile(x, fallback_pct[1])
        mask = (x < lo) | (x > hi)
    else:
        z = 0.6744897501960817 * (x - med) / mad   # ~N(0,1) under normality
        mask = np.abs(z) > mad_z
    x[mask] = np.nan
    return pd.Series(x)

def plot_24_sensors_grid(input_path, output_path, mad_z=3.5,
                         rows=6, cols=4, per_sensor_percentiles=(2, 98)):
    """
    Reads a CSV where column 0 is time/id and the next 24 columns are sensors.
    Creates ONE image with 24 subplots (one per sensor).
    X axis = row index (0..N-1). Y axis = sensor values after outlier removal.
    """
    # Load and coerce to numeric
    df = pd.read_csv(input_path)
    sensors = df.columns[1:]  # assume col 0 is time/id
    sensor_df = df.loc[:, sensors].apply(pd.to_numeric, errors="coerce").round(2)

    # Clean outliers per sensor
    cleaned = sensor_df.apply(lambda s: _remove_outliers_mad(s, mad_z=mad_z))

    # Figure & axes
    n = len(sensors)
    if n != rows * cols:
        # If not exactly 24, adapt grid to fit all sensors
        cols = cols if cols else 4
        rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.2), squeeze=False)
    fig.suptitle(f"Sensors: {os.path.basename(input_path)}", fontsize=16, y=0.995)

    x = np.arange(len(cleaned))

    p_lo, p_hi = per_sensor_percentiles

    for i, col in enumerate(sensors):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        y = cleaned[col].to_numpy(dtype=float)

        # Compute per-sensor y-limits from CLEANED data
        if np.all(np.isnan(y)):
            y_min, y_max = 0, 1
        else:
            y_min = np.nanpercentile(y, p_lo)
            y_max = np.nanpercentile(y, p_hi)
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                y_min, y_max = np.nanmin(y), np.nanmax(y)
                if not np.isfinite(y_min) or not np.isfinite(y_max):
                    y_min, y_max = 0, 1

        ax.plot(x, y, linewidth=0.9)
        ax.set_title(f"{i+1}. {col}", fontsize=10)
        ax.set_xlim(0, len(x)-1)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linewidth=0.4)
        # X label only on bottom row for readability
        if r == rows - 1:
            ax.set_xlabel("Row #")
        # Y label on leftmost column
        if c == 0:
            ax.set_ylabel("Value")

    # Hide any unused axes if sensors < rows*cols
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid to {output_path}")

def process_directory(input_dir, output_dir, mad_z=3.5):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith(".csv"):
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, file.replace(".csv", "_grid.png"))
            plot_24_sensors_grid(in_path, out_path, mad_z=mad_z)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot 24 sensor subplots with outliers removed")
    parser.add_argument("--sensor_dir", required=True, help="Directory with sensor CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the grid images")
    parser.add_argument("--mad_z", type=float, default=3.5, help="MAD z-threshold (lower = stricter)")
    args = parser.parse_args()
    process_directory(args.sensor_dir, args.output_dir, mad_z=args.mad_z)
