import csv
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
    
    # Read all data from source.csv
    source_data = []
    try:
        with open(source_filePath, 'r', newline='') as source_file:
            reader = csv.reader(source_file)
            source_data = list(reader)
    except Exception as e:
        print(f"Error reading {source_filePath}: {e}")
        return
    
    if not source_data:
        print("Error: {source_filePath} is empty!")
        return
    
    print(f"Loaded {len(source_data)} rows from source.csv")
    
    # Initialize data.csv with headers if source has headers
    try:
        with open(data_filePath, 'w', newline='') as data_file:
            writer = csv.writer(data_file)
            # Write header if it exists (assume first row is header)
            if source_data:
                writer.writerow(source_data[0])
    except Exception as e:
        print(f"Error initializing {data_filePath}: {e}")
        return
    
    # Calculate timing for 100Hz (10ms intervals)
    interval = 1.0 / 100.0  # 0.01 seconds
    
    current_row = 1 if len(source_data) > 1 else 0  # Start from second row if header exists
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
            
            # Write current row to data.csv
            try:
                with open(data_filePath, 'a', newline='') as data_file:
                    writer = csv.writer(data_file)
                    writer.writerow(source_data[current_row])
            except Exception as e:
                print(f"Error writing to {data_filePath}: {e}")
                break
            
            # Move to next row (cycle through source data)
            current_row += 1
            if current_row >= len(source_data):
                current_row = 1 if len(source_data) > 1 else 0  # Reset to first data row
                print("End of file reached, restart at beginning")
            
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
    source_fileName = "source.csv"
    data_fileName = "data.csv"
    source_filePath = os.path.join(currentDir, source_fileName)
    data_filePath = os.path.join(currentDir, data_fileName)
    # Start script
    write_csv_at_100hz(source_filePath, data_filePath)
