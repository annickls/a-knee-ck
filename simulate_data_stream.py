import pandas as pd
import time
import os

def write_csv_at_100hz(source_filePath, data_filePath):
    """
    Reads from source.csv and writes to data.csv at 100Hz,
    cycling through the source data continuously.
    """
    
    # Check if source file exists
    if not os.path.exists(source_filePath):
        print(f"Error: {source_filePath} not found!")
        return
    
    # Read all data from source.csv as DataFrame
    try:
        source_df = pd.read_csv(source_filePath, index_col=False)
    except Exception as e:
        print(f"Error reading {source_filePath}: {e}")
        return
    
    if source_df.empty:
        print(f"Error: {source_filePath} is empty!")
        return
    
    print(f"Loaded {len(source_df)} rows from {source_filePath}")

    # Modify the source file for debugging purposes
    source_df.columns = source_df.columns.str.strip()
    source_df[["torque_x","torque_y"]] = 0
    source_df[["torque_z"]] = 10
    # Initialize data.csv with headers
    try:
        # Write just the header to initialize the file
        source_df.iloc[:0].to_csv(data_filePath, index=False)
    except Exception as e:
        print(f"Error initializing {data_filePath}: {e}")
        return
    
    # Calculate timing for 100Hz (10ms intervals)
    interval = 1.0 / 100.0  # 0.01 seconds
    
    current_row = 0  # Start from first row
    start_time = time.time()
    iteration = 0
    
    print("Starting 100Hz data writing... Press Ctrl+C to stop")
    
    try:
        while True:
            # Calculate next target time
            target_time = start_time + (iteration * interval)
            
            # Wait until target time
            current_time = time.time()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Get current row as Series and write to data.csv
            try:
                current_row_data = source_df.iloc[current_row]
                # Convert Series to DataFrame for proper CSV writing
                row_df = pd.DataFrame([current_row_data])
                row_df.to_csv(data_filePath, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error writing to {data_filePath}: {e}")
                break
            
            # Move to next row (cycle through source data)
            current_row += 1
            if current_row >= len(source_df):
                current_row = 0  # Reset to first data row
                print("End of DataFrame reached, restarting at beginning")
            
            iteration += 1
            
            # Print progress every 100 iterations (every second)
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = iteration / elapsed
                print(f"Iteration {iteration}, Rate: {actual_rate:.1f} Hz")
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        actual_rate = iteration / elapsed if elapsed > 0 else 0
        print(f"\nStopped after {iteration} iterations")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Average rate: {actual_rate:.1f} Hz")
        print(f"Data written to {data_filePath}")

if __name__ == "__main__":
    # Set Path to source and data file
    currentDir = os.path.dirname(os.path.abspath(__file__))
    source_fileName = "C:\\Users\\ga94bow\\Documents\\codesandstuff\\knee_analysis\\data_processed\\P1_pre_debug\\var.csv"
    data_fileName = "data.csv"
    source_filePath = os.path.join(currentDir, source_fileName)
    data_filePath = os.path.join(currentDir, data_fileName)
    # Start script
    write_csv_at_100hz(source_filePath, data_filePath)