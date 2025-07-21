import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import re

def extract_start_timestamp(filename):
    """Extract the start timestamp from the filename."""
    # Assuming filename format: timestamp_something.csv
    match = re.search(r'(\d+\.?\d*)', Path(filename).stem)
    if match:
        return float(match.group(1))
    return None

def find_matching_files(directory):
    """Find pairs of CSV files with matching start timestamps."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Group files by start timestamp
    timestamp_groups = {}
    for file in csv_files:
        start_time = extract_start_timestamp(file)
        if start_time is not None:
            if start_time not in timestamp_groups:
                timestamp_groups[start_time] = []
            timestamp_groups[start_time].append(file)
    
    # Find pairs (groups with exactly 2 files)
    pairs = []
    for timestamp, files in timestamp_groups.items():
        if len(files) == 2:
            pairs.append((timestamp, files))
        elif len(files) > 2:
            print(f"Warning: Found {len(files)} files for timestamp {timestamp}: {files}")
    
    return pairs

def identify_file_types(file1, file2):
    """Identify which file has higher frequency based on row count."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if len(df1) > len(df2):
        return file1, file2  # high_freq, low_freq
    else:
        return file2, file1  # high_freq, low_freq

def clean_column_names(df):
    """Clean column names by removing asterisks, hashes, and extra spaces."""
    # Create a mapping of original to cleaned names for debugging
    original_cols = df.columns.tolist()
    
    # Remove asterisks, hashes, and strip whitespace
    cleaned_cols = []
    for col in df.columns:
        cleaned = col.replace('*', '').replace('#', '').strip()
        # Remove any trailing commas that might exist
        cleaned = cleaned.rstrip(',')
        cleaned_cols.append(cleaned)
    
    df.columns = cleaned_cols
    
    # Check for duplicate column names after cleaning
    if len(set(cleaned_cols)) != len(cleaned_cols):
        print("Warning: Duplicate column names found after cleaning!")
        print("Original columns:", original_cols)
        print("Cleaned columns:", cleaned_cols)
        
        # Make column names unique by adding suffix
        seen = {}
        unique_cols = []
        for col in cleaned_cols:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        
        df.columns = unique_cols
        print("Made unique:", unique_cols)
    
    return df

def combine_data_files(high_freq_file, low_freq_file, start_timestamp, output_file):
    """Combine two CSV files, matching timestamps appropriately."""
    
    # Read the data
    print(f"Reading high frequency file: {high_freq_file}")
    print(f"Reading low frequency file: {low_freq_file}")
    
    high_freq_df = pd.read_csv(high_freq_file)
    low_freq_df = pd.read_csv(low_freq_file)
    
    print(f"High frequency file: {high_freq_file} ({len(high_freq_df)} rows)")
    print(f"Low frequency file: {low_freq_file} ({len(low_freq_df)} rows)")
    
    # Show original column names
    print(f"Original high freq columns: {list(high_freq_df.columns)}")
    print(f"Original low freq columns: {list(low_freq_df.columns)}")
    
    # Clean column names
    high_freq_df = clean_column_names(high_freq_df)
    low_freq_df = clean_column_names(low_freq_df)
    
    print(f"Cleaned high freq columns: {list(high_freq_df.columns)}")
    print(f"Cleaned low freq columns: {list(low_freq_df.columns)}")
    print()
    
    # Convert timestamps to absolute time
    # Use the first column as timestamp for both files
    high_freq_timestamp_col = high_freq_df.columns[0]
    low_freq_timestamp_col = low_freq_df.columns[0]
    
    print(f"Using timestamp columns: high_freq='{high_freq_timestamp_col}', low_freq='{low_freq_timestamp_col}'")
    
    # Ensure timestamp columns are numeric
    high_freq_df[high_freq_timestamp_col] = pd.to_numeric(high_freq_df[high_freq_timestamp_col], errors='coerce')
    low_freq_df[low_freq_timestamp_col] = pd.to_numeric(low_freq_df[low_freq_timestamp_col], errors='coerce')
    
    # Convert timestamps to absolute time
    high_freq_df['abs_timestamp'] = high_freq_df[high_freq_timestamp_col] + start_timestamp
    low_freq_df['abs_timestamp'] = low_freq_df[low_freq_timestamp_col] + start_timestamp
    
    # Sort by timestamp
    high_freq_df = high_freq_df.sort_values('abs_timestamp')
    low_freq_df = low_freq_df.sort_values('abs_timestamp')
    
    # Start with low frequency data as the base
    combined_df = low_freq_df.copy()
    
    # Get all columns except timestamp and abs_timestamp
    low_freq_data_cols = [col for col in low_freq_df.columns if col not in [low_freq_timestamp_col, 'abs_timestamp']]
    high_freq_data_cols = [col for col in high_freq_df.columns if col not in [high_freq_timestamp_col, 'abs_timestamp']]
    
    print(f"Low freq data columns ({len(low_freq_data_cols)}): {low_freq_data_cols}")
    print(f"High freq data columns ({len(high_freq_data_cols)}): {high_freq_data_cols}")
    
    # Check for column name conflicts
    conflicting_cols = set(low_freq_data_cols) & set(high_freq_data_cols)
    if conflicting_cols:
        print(f"Warning: Found conflicting column names: {conflicting_cols}")
        # Rename conflicting columns in high frequency data
        for col in conflicting_cols:
            new_col_name = f"{col}_highfreq"
            high_freq_df = high_freq_df.rename(columns={col: new_col_name})
            # Update the list
            high_freq_data_cols = [new_col_name if c == col else c for c in high_freq_data_cols]
            print(f"Renamed '{col}' to '{new_col_name}' in high frequency data")
    
    # Add high frequency columns to the combined dataframe
    for col in high_freq_data_cols:
        combined_df[col] = np.nan
    
    print(f"Adding {len(high_freq_data_cols)} high frequency columns to the combined data")
    
    # For each row in low frequency data, find the closest timestamp in high frequency data
    for idx, row in combined_df.iterrows():
        target_timestamp = row['abs_timestamp']
        
        # Find the closest timestamp in high frequency data
        time_diffs = np.abs(high_freq_df['abs_timestamp'] - target_timestamp)
        closest_idx = time_diffs.idxmin()
        
        # Copy all high frequency data columns to the combined dataframe
        for col in high_freq_data_cols:
            combined_df.at[idx, col] = high_freq_df.at[closest_idx, col]
    
    # Remove the absolute timestamp column
    combined_df = combined_df.drop('abs_timestamp', axis=1)
    
    # Reorder columns: timestamp first, then low freq data, then high freq data
    final_column_order = [low_freq_timestamp_col] + low_freq_data_cols + high_freq_data_cols
    combined_df = combined_df[final_column_order]
    
    # Verify all columns are present
    print(f"Final combined dataframe columns: {list(combined_df.columns)}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Final shape: {combined_df.shape}")
    
    # Save the combined data
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to: {output_file}")
    
    return combined_df

def main():
    """Main function to process all file pairs in a directory."""
    
    # Configuration
    input_directory = "."  # Current directory - change as needed
    output_directory = "./combined_data"  # Output directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find matching file pairs
    file_pairs = find_matching_files(input_directory)
    
    if not file_pairs:
        print("No matching file pairs found!")
        return
    
    print(f"Found {len(file_pairs)} file pairs to process:")
    
    # Process each pair
    for start_timestamp, files in file_pairs:
        print(f"\nProcessing files with start timestamp: {start_timestamp}")
        
        # Identify which file has higher frequency
        high_freq_file, low_freq_file = identify_file_types(files[0], files[1])
        
        # Generate output filename
        output_filename = f"combined_{start_timestamp}.csv"
        output_path = os.path.join(output_directory, output_filename)
        
        # Combine the files
        try:
            combined_df = combine_data_files(high_freq_file, low_freq_file, 
                                           start_timestamp, output_path)
            print(f"Successfully combined {len(combined_df)} rows")
        except Exception as e:
            print(f"Error processing files: {e}")
            import traceback
            traceback.print_exc()

# Alternative function for processing specific files
def combine_specific_files(high_freq_file, low_freq_file, output_file):
    """Combine two specific CSV files."""
    
    # Extract start timestamp from filename
    start_timestamp = extract_start_timestamp(high_freq_file)
    if start_timestamp is None:
        start_timestamp = extract_start_timestamp(low_freq_file)
    
    if start_timestamp is None:
        print("Could not extract start timestamp from filenames!")
        return None
    
    return combine_data_files(high_freq_file, low_freq_file, start_timestamp, output_file)

if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Process all matching pairs in directory
    # main()
    
    # Option 2: Process specific files (uncomment and modify as needed)
    combine_specific_files("data_20250711_150014.csv", "20250711_152806_rot_2.csv", "combined_output.csv")