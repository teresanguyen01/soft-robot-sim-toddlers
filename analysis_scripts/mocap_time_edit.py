import pandas as pd
import os

def time_edit(file_path, output_folder):
    """
    This script reads a CSV file containing motion capture data, calculates the time in milliseconds for each frame,
    and adds this as a new column to the DataFrame. It then saves the updated DataFrame to a specified output folder.
    """
    mocap_data = pd.read_csv(file_path)
    mocap_data['time_ms'] = mocap_data['Frame'] * (1000 / 100)
    
    # Generate output file name based on input file name
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_with_time.csv")
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    mocap_data.to_csv(output_file, index=False)
    print(f"Processed file saved as '{output_file}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add time in milliseconds to motion capture data.")
    parser.add_argument('--file', required=True, help='Path to the input CSV file containing motion capture data.')
    parser.add_argument('--output_dir', required=True, help='Path to the folder where the output file will be saved.')
    args = parser.parse_args()

    time_edit(args.file, args.output_dir)