import pandas as pd
import time
import os
from datetime import datetime

def update_csv_at_50hz(input_file='data4.csv', output_file='data.csv'):
    """
    Updates and saves a CSV file with a new line at approximately 50 Hz (every 0.02 seconds).
    The function reads data from the input_file and continuously appends rows to the output_file.
    
    Args:
        input_file (str): Path to the source CSV file
        output_file (str): Path to the target CSV file that will be updated
    """
    try:
        # Read the input CSV file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        
        source_data = pd.read_csv(input_file)
        
        # Check if 'timestamp' exists in source data
        if 'timestamp' not in source_data.columns:
            # If not, add it as a new column
            source_data['timestamp'] = None
        
        # If output file doesn't exist, create it with headers
        if not os.path.exists(output_file):
            # Ensure 'timestamp' column exists in the output headers
            headers = list(source_data.columns)
            if 'timestamp' not in headers:
                source_data['timestamp'] = None
            source_data.head(0).to_csv(output_file, index=False)
            print(f"Created new output file: {output_file}")
        
        row_index = 0
        total_rows = len(source_data)
        
        print(f"Starting to update {output_file} at 50 Hz using data from {input_file}")
        print(f"Total rows in source data: {total_rows}")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                # Get current row from source data (cycling through all rows)
                current_row = source_data.iloc[row_index % total_rows].copy()
                
                # Create a new dictionary from the current row
                row_dict = current_row.to_dict()
                
                # Add timestamp to the dictionary instead of directly to the Series
                row_dict['timestamp'] = datetime.now().strftime('%H%M%S.%f')
                
                # Convert the dictionary to a DataFrame (avoids the dtype warning)
                row_df = pd.DataFrame([row_dict])
                
                # Append the row to the output file
                row_df.to_csv(output_file, mode='a', header=False, index=False)
                
                # Move to next row
                row_index += 1
                
                # Print status occasionally
                #if row_index % 100 == 0:
                #    print(f"Added {row_index} rows to {output_file}")
                
                # Sleep to maintain approximately 50 Hz (0.02 seconds per iteration)
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Added {row_index} rows to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_csv_at_50hz()