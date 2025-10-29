import pandas as pd
import time
import os
import threading
import sys

# Windows-compatible keyboard input handler
if sys.platform == 'win32':
    import msvcrt
    
    def get_key():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\xe0':  # Special key prefix on Windows
                key = msvcrt.getch()
                if key == b'K':  # Left arrow
                    return 'LEFT'
                elif key == b'M':  # Right arrow
                    return 'RIGHT'
            elif key == b'p' or key == b'P':
                return 'PAUSE'
        return None
else:
    # Unix-like systems (placeholder)
    def get_key():
        return None

# Global control variables
paused = False
skip_request = None  # 'FORWARD' or 'BACKWARD'
control_lock = threading.Lock()

def keyboard_handler():
    """Handle keyboard input in a separate thread"""
    global paused, skip_request
    
    while True:
        key = get_key()
        if key:
            with control_lock:
                if key == 'PAUSE':
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n[{status}] Press 'p' to toggle, left/right arrows to skip")
                elif key == 'LEFT':
                    skip_request = 'BACKWARD'
                elif key == 'RIGHT':
                    skip_request = 'FORWARD'
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage

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

    source_df.columns = source_df.columns.str.strip()

    # Modify the source file for debugging purposes
    # source_df["tibia_x"] = source_df["tibia_x"][0]
    # source_df["tibia_y"] = source_df["tibia_y"][0]
    # source_df["tibia_z"] = source_df["tibia_z"][0]
    # source_df["tibia_qx"] = source_df["tibia_qx"][0]
    # source_df["tibia_qy"] = source_df["tibia_qy"][0]
    # source_df["tibia_qz"] = source_df["tibia_qz"][0]
    # source_df["tibia_qw"] = source_df["tibia_qw"][0]
    # Initialize data.csv with headers
    try:
        # Write just the header to initialize the file
        source_df.iloc[:0].to_csv(data_filePath, index=False)
    except Exception as e:
        print(f"Error initializing {data_filePath}: {e}")
        return
    
    # Calculate timing for 100Hz (10ms intervals)
    interval = 1.0 / 100.0  # 0.01 seconds
    
    global paused, skip_request
    
    current_row = 0  # Start from first row
    start_time = time.time()
    pause_start_time = None
    total_paused_time = 0.0
    iteration = 0
    
    # Start keyboard handler thread
    keyboard_thread = threading.Thread(target=keyboard_handler, daemon=True)
    keyboard_thread.start()
    
    print("Starting 100Hz data writing...")
    print("Controls: 'p' = pause/resume, Left arrow = skip back 200 rows, Right arrow = skip forward 200 rows")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Handle pause state changes and timing
            with control_lock:
                # Track pause/resume timing
                if paused and pause_start_time is None:
                    # Just entered pause state
                    pause_start_time = time.time()
                elif not paused and pause_start_time is not None:
                    # Just resumed from pause state
                    total_paused_time += time.time() - pause_start_time
                    pause_start_time = None
                
                # Handle skip requests
                if skip_request == 'FORWARD':
                    current_row = (current_row + 200) % len(source_df)
                    skip_request = None
                    print(f"Skipped forward 200 rows to row {current_row}")
                    
                    # If paused, write single row and stay paused
                    if paused:
                        try:
                            current_row_data = source_df.iloc[current_row]
                            row_df = pd.DataFrame([current_row_data])
                            row_df.to_csv(data_filePath, mode='a', header=False, index=False)
                        except Exception as e:
                            print(f"Error writing to {data_filePath}: {e}")
                        continue
                        
                elif skip_request == 'BACKWARD':
                    current_row = (current_row - 200) % len(source_df)
                    skip_request = None
                    print(f"Skipped backward 200 rows to row {current_row}")
                    
                    # If paused, write single row and stay paused
                    if paused:
                        try:
                            current_row_data = source_df.iloc[current_row]
                            row_df = pd.DataFrame([current_row_data])
                            row_df.to_csv(data_filePath, mode='a', header=False, index=False)
                        except Exception as e:
                            print(f"Error writing to {data_filePath}: {e}")
                        continue
                
                # If paused and no skip request, just wait
                if paused:
                    time.sleep(0.01)
                    continue
            
            # Calculate next target time (adjusted for total paused time)
            adjusted_start_time = start_time + total_paused_time
            target_time = adjusted_start_time + (iteration * interval)
            
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
                print("End of DataFrame reached, restarting at beginning") if len(source_df)>50 else None
            
            iteration += 1
            
            # Print progress every 100 iterations (every second)
            if iteration % 100 == 0:
                # Calculate elapsed time excluding paused time
                current_paused_time = total_paused_time
                if pause_start_time is not None:  # Currently paused
                    current_paused_time += time.time() - pause_start_time
                elapsed_active = time.time() - start_time - current_paused_time
                actual_rate = iteration / elapsed_active if elapsed_active > 0 else 0
                print(f"Iteration {iteration}, Rate: {actual_rate:.1f} Hz, Row: {current_row}")
    
    except KeyboardInterrupt:
        # Calculate final statistics excluding paused time
        current_paused_time = total_paused_time
        if pause_start_time is not None:  # Was paused when stopped
            current_paused_time += time.time() - pause_start_time
        elapsed_total = time.time() - start_time
        elapsed_active = elapsed_total - current_paused_time
        actual_rate = iteration / elapsed_active if elapsed_active > 0 else 0
        print(f"\nStopped after {iteration} iterations")
        print(f"Total elapsed time: {elapsed_total:.2f} seconds (paused: {current_paused_time:.2f}s)")
        print(f"Active time: {elapsed_active:.2f} seconds")
        print(f"Average rate: {actual_rate:.1f} Hz")
        print(f"Data written to {data_filePath}")

if __name__ == "__main__":
    # Set Path to source and data folder
    currentDir = os.path.dirname(os.path.abspath(__file__))
    sourceDir = "C:\\Users\\ga94bow\\Documents\\codesandstuff\\knee_analysis\\data_processed"
    source_fileName = os.path.join("neutral.csv")

    # Set Patients name and examiner if needed
    patientName = "P4_pre"
    examinerName = "Claudio"

    # Set name of the test
    testName = "neutral.csv"
    output_fileName = "data.csv"

    # Combine Paths; If debug is set, then take the measuremung where the leg is supposed to be straight
    if examinerName == "debug":
        source_filePath = os.path.join(currentDir, "data_for_gui", patientName, "streckung_GUI.csv")
    else:
        source_filePath = os.path.join(sourceDir, patientName, examinerName, testName)
    output_filePath = os.path.join(currentDir, output_fileName)
    # Start script
    write_csv_at_100hz(source_filePath, output_filePath)