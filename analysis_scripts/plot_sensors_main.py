import os
import pandas as pd
import matplotlib.pyplot as plt

# Custom labels for plotting (only used in titles)
SENSOR_NAMES = [
    "1. right elbow", "2. right shoulder", "3. right collarbone", "4. bottom back", "5. top back",
    "6. left collarbone", "7. left shoulder", "8. left armpit (back)", "9. left elbow", "10. left elbow (back)",
    "11. left armpit (front)", "12. waist left a", "13. chest l", "14. stomach l", "15. hip right",
    "16. waist left b", "17. right armpit (back)", "18. right elbow (front)", "19. right armpit (front)",
    "20. waist right a", "21. waist right b", "22. hip left", "23. stomach r", "24. chest r"
]

def plot_24_sensors(csv_path, output_path):
    df = pd.read_csv(csv_path)

    if df.shape[1] != 25:
        print(f"Skipping {csv_path}: Expected 25 columns (24 sensors + 1 time_ms), got {df.shape[1]}")
        return

    time = df.iloc[:, -1].to_numpy()  # Use the last column as time_ms
    sensor_cols = df.columns[:-1]  # All columns except the last one are sensor data

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 15), sharex=True)
    fig.suptitle(f"Sensors: {os.path.basename(csv_path)}", fontsize=18)

    for idx, (col, label) in enumerate(zip(sensor_cols, SENSOR_NAMES)):
        row = idx // 4
        col_idx = idx % 4
        ax = axes[row, col_idx]
        ax.plot(time, df[col].to_numpy(), linewidth=1.0)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Value")
        if row == 5:
            ax.set_xlabel("Time (ms)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
    
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, file.replace(".csv", ".png"))
            plot_24_sensors(in_path, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot 24 sensor columns in subplots for all CSVs in a directory")
    parser.add_argument("--sensor_dir", help="Directory with 25-column CSV files (Time_ms + 24 sensors)")
    parser.add_argument("--output_dir", help="Directory to save .png plots")
    args = parser.parse_args()

    process_directory(args.sensor_dir, args.output_dir)