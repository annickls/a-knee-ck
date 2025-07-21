import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ADJUSTABLE PARAMETERS
MOVING_AVERAGE_WINDOW = 15  # Adjust this value to change the moving average window size
FLEXION_STEP = 0.5  # Step size for flexion bins

print("Current working directory:", os.getcwd())
print("Changed working directory to:", os.getcwd())

# Read the CSV files for LEFT PLOTS (new dataset)
df1_left = pd.read_csv('20250711_134115_neutral_1.csv')
df2_left = pd.read_csv('20250711_135916_neutral_2.csv')
df3_left = pd.read_csv('20250711_141503_maybe_neutral_3.csv')

# Add dataset identifiers for left plots
df1_left['dataset'] = 'neutral_1'
df2_left['dataset'] = 'neutral_2'
df3_left['dataset'] = 'neutral_3'

# Combine datasets for LEFT PLOTS
dfneutral = pd.read_csv('neutral_p2_all.csv')
dfneutral['dataset'] = 'neutral'
df_left = pd.concat([df1_left, df2_left, df3_left], ignore_index=True)

# Read the CSV files for RIGHT PLOTS (original dataset)
df1_right = pd.read_csv('20250711_151100_neutral_1.csv')
df2_right = pd.read_csv('20250711_152351_neutral_2.csv')
df3_right = pd.read_csv('20250711_153725_neutral_3.csv')

# Add dataset identifiers for right plots
df1_right['dataset'] = 'neutral_1'
df2_right['dataset'] = 'neutral_2'
df3_right['dataset'] = 'neutral_3'

# Combine datasets for RIGHT PLOTS
df_right = pd.concat([df1_right, df2_right, df3_right], ignore_index=True)

# Clean column names by stripping whitespace for both datasets
df_left.columns = df_left.columns.str.strip()
df_right.columns = df_right.columns.str.strip()

# Filter ranges
filter_range_narrow = 300  # For left plots 
filter_range_wide = 300    # For right plots 

# Print available columns for debugging
print("Available columns in LEFT dataset:")
print(df_left.columns.tolist())
print("Available columns in RIGHT dataset:")
print(df_right.columns.tolist())
print(f"\nLEFT dataset shape: {df_left.shape}")
print(f"RIGHT dataset shape: {df_right.shape}")
print(f"LEFT dataset counts: {df_left['dataset'].value_counts()}")
print(f"RIGHT dataset counts: {df_right['dataset'].value_counts()}")

# Check if required columns exist and map them correctly for both datasets
required_columns = ['Flexion', 'Rotation', 'Adduction', 'Anterior_Posterior', 'Medial_Lateral', 'Tx', 'Ty', 'Tz']

def get_column_mapping(df, dataset_name):
    column_mapping = {}
    for col in required_columns:
        if col in df.columns:
            column_mapping[col] = col
        else:
            # Try to find similar column names (case insensitive)
            similar_cols = [c for c in df.columns if col.lower() in c.lower()]
            if similar_cols:
                column_mapping[col] = similar_cols[0]
                print(f"{dataset_name}: Using '{similar_cols[0]}' for '{col}'")
            else:
                print(f"Warning: Column '{col}' not found in {dataset_name}")
    return column_mapping

column_mapping_left = get_column_mapping(df_left, "LEFT dataset")
column_mapping_right = get_column_mapping(df_right, "RIGHT dataset")

# Check if we have all required columns for both datasets
missing_cols_left = [col for col in required_columns if col not in column_mapping_left]
missing_cols_right = [col for col in required_columns if col not in column_mapping_right]

if missing_cols_left:
    print(f"Missing columns in LEFT dataset: {missing_cols_left}")
if missing_cols_right:
    print(f"Missing columns in RIGHT dataset: {missing_cols_right}")

if missing_cols_left or missing_cols_right:
    print("Please check your CSV files and column names")
    exit()

# Calculate total torque for both datasets
df_left['Total_Torque'] = df_left[column_mapping_left['Tx']] + df_left[column_mapping_left['Ty']] + df_left[column_mapping_left['Tz']]
df_right['Total_Torque'] = df_right[column_mapping_right['Tx']] + df_right[column_mapping_right['Ty']] + df_right[column_mapping_right['Tz']]

# Create filtered datasets for LEFT plots
df_rot_left = df_left[(df_left[column_mapping_left['Adduction']] > -filter_range_narrow) & 
                      (df_left[column_mapping_left['Adduction']] < filter_range_narrow)]
df_add_left = df_left[(df_left[column_mapping_left['Rotation']] > -filter_range_narrow) & 
                      (df_left[column_mapping_left['Rotation']] < filter_range_narrow)]
df_ap_left = df_left[(df_left[column_mapping_left['Medial_Lateral']] > -filter_range_narrow) & 
                     (df_left[column_mapping_left['Medial_Lateral']] < filter_range_narrow)]
df_ml_left = df_left[(df_left[column_mapping_left['Anterior_Posterior']] > -filter_range_narrow) & 
                     (df_left[column_mapping_left['Anterior_Posterior']] < filter_range_narrow)]

# Create filtered datasets for RIGHT plots
df_rot_right = df_right[(df_right[column_mapping_right['Adduction']] > -filter_range_wide) & 
                        (df_right[column_mapping_right['Adduction']] < filter_range_wide)]
df_add_right = df_right[(df_right[column_mapping_right['Rotation']] > -filter_range_wide) & 
                        (df_right[column_mapping_right['Rotation']] < filter_range_wide)]
df_ap_right = df_right[(df_right[column_mapping_right['Medial_Lateral']] > -filter_range_wide) & 
                       (df_right[column_mapping_right['Medial_Lateral']] < filter_range_wide)]
df_ml_right = df_right[(df_right[column_mapping_right['Anterior_Posterior']] > -filter_range_wide) & 
                       (df_right[column_mapping_right['Anterior_Posterior']] < filter_range_wide)]

print(f"\nFiltered dataset sizes:")
print(f"LEFT plots ({filter_range_narrow}°):")
print(f"  Rotation plot: {len(df_rot_left)} points")
print(f"  Adduction plot: {len(df_add_left)} points")
print(f"  Anterior_Posterior plot: {len(df_ap_left)} points")
print(f"  Medial_Lateral plot: {len(df_ml_left)} points")
print(f"RIGHT plots ({filter_range_wide}°):")
print(f"  Rotation plot: {len(df_rot_right)} points")
print(f"  Adduction plot: {len(df_add_right)} points")
print(f"  Anterior_Posterior plot: {len(df_ap_right)} points")
print(f"  Medial_Lateral plot: {len(df_ml_right)} points")

# Function to calculate averaged data in flexion steps
def calculate_averaged_data(df, flexion_col, value_col, step=FLEXION_STEP):
    """
    Calculate average and standard deviation for given flexion steps
    
    Parameters:
    df: DataFrame with the data
    flexion_col: name of the flexion column
    value_col: name of the value column to average
    step: step size in degrees
    
    Returns:
    flexion_bins: array of flexion values
    avg_values: array of averaged values
    std_values: array of standard deviations
    """
    
    # Define flexion range
    min_flex = df[flexion_col].min()
    max_flex = df[flexion_col].max()
    
    # Create bins for flexion
    flexion_bins = np.arange(min_flex, max_flex + step, step)
    
    avg_values = []
    std_values = []
    valid_bins = []
    
    for i in range(len(flexion_bins) - 1):
        # Get data points in this flexion bin
        mask = (df[flexion_col] >= flexion_bins[i]) & (df[flexion_col] < flexion_bins[i + 1])
        bin_data = df[mask][value_col]
        
        if len(bin_data) > 0:  # Only include bins with data
            avg_values.append(bin_data.mean())
            std_values.append(bin_data.std())
            valid_bins.append(flexion_bins[i] + step/2)  # Use center of bin
    
    return np.array(valid_bins), np.array(avg_values), np.array(std_values)

# Function to calculate moving average
def calculate_moving_average(values, window=MOVING_AVERAGE_WINDOW):
    """
    Calculate moving average with specified window size
    
    Parameters:
    values: array of values
    window: window size for moving average
    
    Returns:
    moving_avg: array of moving averages
    """
    if len(values) < window:
        return values  # Return original if not enough data points
    
    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
    return moving_avg

# Calculate averaged data for all plots
print(f"\nCalculating averaged data with moving average window = {MOVING_AVERAGE_WINDOW}...")

# LEFT plots
flexion_rot_left, avg_rot_left, std_rot_left = calculate_averaged_data(
    df_rot_left, column_mapping_left['Flexion'], column_mapping_left['Rotation'])
flexion_add_left, avg_add_left, std_add_left = calculate_averaged_data(
    df_add_left, column_mapping_left['Flexion'], column_mapping_left['Adduction'])
flexion_ap_left, avg_ap_left, std_ap_left = calculate_averaged_data(
    df_ap_left, column_mapping_left['Flexion'], column_mapping_left['Anterior_Posterior'])
flexion_ml_left, avg_ml_left, std_ml_left = calculate_averaged_data(
    df_ml_left, column_mapping_left['Flexion'], column_mapping_left['Medial_Lateral'])

# RIGHT plots
flexion_rot_right, avg_rot_right, std_rot_right = calculate_averaged_data(
    df_rot_right, column_mapping_right['Flexion'], column_mapping_right['Rotation'])
flexion_add_right, avg_add_right, std_add_right = calculate_averaged_data(
    df_add_right, column_mapping_right['Flexion'], column_mapping_right['Adduction'])
flexion_ap_right, avg_ap_right, std_ap_right = calculate_averaged_data(
    df_ap_right, column_mapping_right['Flexion'], column_mapping_right['Anterior_Posterior'])
flexion_ml_right, avg_ml_right, std_ml_right = calculate_averaged_data(
    df_ml_right, column_mapping_right['Flexion'], column_mapping_right['Medial_Lateral'])

# Calculate moving averages
# LEFT plots
ma_rot_left = calculate_moving_average(avg_rot_left)
ma_add_left = calculate_moving_average(avg_add_left)
ma_ap_left = calculate_moving_average(avg_ap_left)
ma_ml_left = calculate_moving_average(avg_ml_left)

# RIGHT plots
ma_rot_right = calculate_moving_average(avg_rot_right)
ma_add_right = calculate_moving_average(avg_add_right)
ma_ap_right = calculate_moving_average(avg_ap_right)
ma_ml_right = calculate_moving_average(avg_ml_right)

# Adjust flexion arrays for moving average (they will be shorter)
def adjust_flexion_for_ma(flexion_array, ma_array):
    if len(ma_array) < len(flexion_array):
        # Center the moving average data
        start_idx = (len(flexion_array) - len(ma_array)) // 2
        return flexion_array[start_idx:start_idx + len(ma_array)]
    return flexion_array

# Adjusted flexion arrays for moving averages
flexion_ma_rot_left = adjust_flexion_for_ma(flexion_rot_left, ma_rot_left)
flexion_ma_add_left = adjust_flexion_for_ma(flexion_add_left, ma_add_left)
flexion_ma_ap_left = adjust_flexion_for_ma(flexion_ap_left, ma_ap_left)
flexion_ma_ml_left = adjust_flexion_for_ma(flexion_ml_left, ma_ml_left)

flexion_ma_rot_right = adjust_flexion_for_ma(flexion_rot_right, ma_rot_right)
flexion_ma_add_right = adjust_flexion_for_ma(flexion_add_right, ma_add_right)
flexion_ma_ap_right = adjust_flexion_for_ma(flexion_ap_right, ma_ap_right)
flexion_ma_ml_right = adjust_flexion_for_ma(flexion_ml_right, ma_ml_right)

# Create a figure with 8 subplots (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(20, 20))
fig.suptitle(f'Averaged Data P2 vs P6 ({FLEXION_STEP}° Flexion Steps, MA Window={MOVING_AVERAGE_WINDOW})', 
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Rotation vs Flexion (swapped axes)
# Left (P2)
axes[0, 0].plot(avg_rot_left, flexion_rot_left, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[0, 0].plot(avg_rot_left + std_rot_left, flexion_rot_left, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[0, 0].plot(avg_rot_left - std_rot_left, flexion_rot_left, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[0, 0].plot(ma_rot_left, flexion_ma_rot_left, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[0, 0].fill_betweenx(flexion_rot_left, avg_rot_left - std_rot_left, avg_rot_left + std_rot_left, 
                        alpha=0.2, color='blue')
axes[0, 0].set_ylabel('Flexion (°)', fontsize=10)
axes[0, 0].set_xlabel('Internal Rotation (°) ← → External Rotation (°)', fontsize=10)
axes[0, 0].set_title('Rotation - P2', fontsize=12, pad=15)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].set_ylim(-10, 120)
axes[0, 0].set_xlim(-40, 40)
axes[0, 0].invert_yaxis()

# Right (P6)
axes[0, 2].plot(avg_rot_right, flexion_rot_right, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[0, 2].plot(avg_rot_right + std_rot_right, flexion_rot_right, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[0, 2].plot(avg_rot_right - std_rot_right, flexion_rot_right, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[0, 2].plot(ma_rot_right, flexion_ma_rot_right, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[0, 2].fill_betweenx(flexion_rot_right, avg_rot_right - std_rot_right, avg_rot_right + std_rot_right, 
                        alpha=0.2, color='blue')
axes[0, 2].set_ylabel('Flexion (°)', fontsize=10)
axes[0, 2].set_xlabel('Internal Rotation (°) ← → External Rotation (°)', fontsize=10)
axes[0, 2].set_title('Rotation - P6', fontsize=12, pad=15)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()
axes[0, 2].set_ylim(-10, 120)
axes[0, 2].set_xlim(-40, 40)
axes[0, 2].invert_yaxis()

# Plot 2: Adduction vs Flexion (swapped axes)
# Left (P2)
axes[0, 1].plot(avg_add_left, flexion_add_left, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[0, 1].plot(avg_add_left + std_add_left, flexion_add_left, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[0, 1].plot(avg_add_left - std_add_left, flexion_add_left, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[0, 1].plot(ma_add_left, flexion_ma_add_left, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[0, 1].fill_betweenx(flexion_add_left, avg_add_left - std_add_left, avg_add_left + std_add_left, 
                        alpha=0.2, color='blue')
axes[0, 1].set_ylabel('Flexion (°)', fontsize=10)
axes[0, 1].set_xlabel('Varus (°) ← → Valgus (°)', fontsize=10)
axes[0, 1].set_title('Adduction/Abduction - P2', fontsize=12, pad=15)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_ylim(-10, 120)
axes[0, 1].set_xlim(-20, 20)
axes[0, 1].invert_yaxis()

# Right (P6)
axes[0, 3].plot(avg_add_right, flexion_add_right, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[0, 3].plot(avg_add_right + std_add_right, flexion_add_right, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[0, 3].plot(avg_add_right - std_add_right, flexion_add_right, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[0, 3].plot(ma_add_right, flexion_ma_add_right, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[0, 3].fill_betweenx(flexion_add_right, avg_add_right - std_add_right, avg_add_right + std_add_right, 
                        alpha=0.2, color='blue')
axes[0, 3].set_ylabel('Flexion (°)', fontsize=10)
axes[0, 3].set_xlabel('Varus (°) ← → Valgus (°)', fontsize=10)
axes[0, 3].set_title('Adduction/Abduction - P6', fontsize=12, pad=15)
axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].legend()
axes[0, 3].set_ylim(-10, 120)
axes[0, 3].set_xlim(-20, 20)
axes[0, 3].invert_yaxis()

# Plot 3: Anterior_Posterior vs Flexion (swapped axes)
# Left (P2)
axes[1, 0].plot(avg_ap_left, flexion_ap_left, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[1, 0].plot(avg_ap_left + std_ap_left, flexion_ap_left, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[1, 0].plot(avg_ap_left - std_ap_left, flexion_ap_left, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[1, 0].plot(ma_ap_left, flexion_ma_ap_left, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[1, 0].fill_betweenx(flexion_ap_left, avg_ap_left - std_ap_left, avg_ap_left + std_ap_left, 
                        alpha=0.2, color='blue')
axes[1, 0].set_ylabel('Flexion (°)', fontsize=10)
axes[1, 0].set_xlabel('Posterior (mm) ← → Anterior (mm)', fontsize=10)
axes[1, 0].set_title('Anterior-Posterior - P2', fontsize=12, pad=15)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].set_ylim(-10, 120)
axes[1, 0].set_xlim(-20, 20)
axes[1, 0].invert_yaxis()

# Right (P6)
axes[1, 2].plot(avg_ap_right, flexion_ap_right, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[1, 2].plot(avg_ap_right + std_ap_right, flexion_ap_right, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[1, 2].plot(avg_ap_right - std_ap_right, flexion_ap_right, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[1, 2].plot(ma_ap_right, flexion_ma_ap_right, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[1, 2].fill_betweenx(flexion_ap_right, avg_ap_right - std_ap_right, avg_ap_right + std_ap_right, 
                        alpha=0.2, color='blue')
axes[1, 2].set_ylabel('Flexion (°)', fontsize=10)
axes[1, 2].set_xlabel('Posterior (mm) ← → Anterior (mm)', fontsize=10)
axes[1, 2].set_title('Anterior-Posterior - P6', fontsize=12, pad=15)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()
axes[1, 2].set_ylim(-10, 120)
axes[1, 2].set_xlim(-20, 20)
axes[1, 2].invert_yaxis()

# Plot 4: Medial_Lateral vs Flexion (swapped axes)
# Left (P2)
axes[1, 1].plot(avg_ml_left, flexion_ml_left, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[1, 1].plot(avg_ml_left + std_ml_left, flexion_ml_left, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[1, 1].plot(avg_ml_left - std_ml_left, flexion_ml_left, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[1, 1].plot(ma_ml_left, flexion_ma_ml_left, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[1, 1].fill_betweenx(flexion_ml_left, avg_ml_left - std_ml_left, avg_ml_left + std_ml_left, 
                        alpha=0.2, color='blue')
axes[1, 1].set_ylabel('Flexion (°)', fontsize=10)
axes[1, 1].set_xlabel('Lateral (mm) ← → Medial (mm)', fontsize=10)
axes[1, 1].set_title('Medial-Lateral - P2', fontsize=12, pad=15)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].set_ylim(-10, 120)
axes[1, 1].set_xlim(-50, 50)
axes[1, 1].invert_yaxis()

# Right (P6)
axes[1, 3].plot(avg_ml_right, flexion_ml_right, 'b-', linewidth=2, alpha=0.5, label='Average')
axes[1, 3].plot(avg_ml_right + std_ml_right, flexion_ml_right, 'b--', linewidth=1, alpha=0.3, label='+1 SD')
axes[1, 3].plot(avg_ml_right - std_ml_right, flexion_ml_right, 'b--', linewidth=1, alpha=0.3, label='-1 SD')
axes[1, 3].plot(ma_ml_right, flexion_ma_ml_right, 'r-', linewidth=3, label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})')
axes[1, 3].fill_betweenx(flexion_ml_right, avg_ml_right - std_ml_right, avg_ml_right + std_ml_right, 
                        alpha=0.2, color='blue')
axes[1, 3].set_ylabel('Flexion (°)', fontsize=10)
axes[1, 3].set_xlabel('Lateral (mm) ← → Medial (mm)', fontsize=10)
axes[1, 3].set_title('Medial-Lateral - P6', fontsize=12, pad=15)
axes[1, 3].grid(True, alpha=0.3)
axes[1, 3].legend()
axes[1, 3].set_ylim(-10, 120)
axes[1, 3].set_xlim(-50, 50)
axes[1, 3].invert_yaxis()

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Show the plots
plt.show()