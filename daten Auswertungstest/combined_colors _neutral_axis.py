import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Current working directory:", os.getcwd())
#os.chdir(r'C:\files_Annick\Studium Unterlagen\Master\masterarbeit\auswertung\P2 and P6')
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
#df_left = pd.concat([df1_left, df2_left, df3_left, df4_left, df5_left, df6_left, df7_left, 
 #                    df8_left, df9_left, df10_left, df12_left, df13_left, df14_left, df15_left], 
  #                  ignore_index=True)
dfneutral = pd.read_csv('neutral_p2_all.csv')
dfneutral['dataset'] = 'neutral'
df_left = pd.concat([df1_left, df2_left, df3_left], 
                    ignore_index=True)

# Read the CSV files for RIGHT PLOTS (original dataset)
df1_right = pd.read_csv('20250711_151100_neutral_1.csv')
df2_right = pd.read_csv('20250711_152351_neutral_2.csv')
df3_right = pd.read_csv('20250711_153725_neutral_3.csv')


# Add dataset identifiers for right plots
df1_right['dataset'] = 'neutral_1'
df2_right['dataset'] = 'neutral_2'
df3_right['dataset'] = 'neutral_3'


# Combine datasets for RIGHT PLOTS
df_right = pd.concat([df1_right, df2_right, df3_right], 
                     ignore_index=True)

# Clean column names by stripping whitespace for both datasets
df_left.columns = df_left.columns.str.strip()
df_right.columns = df_right.columns.str.strip()

# Filter ranges
filter_range_narrow = 300 # For left plots 
filter_range_wide = 300     # For right plots 

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

# Function to assign colors based on torque
def assign_torque_colors(torque_values):
    colors = []
    for torque in torque_values:
        # Determine the 1Nm window
        torque_window = int(np.floor(torque))
        
        if torque >= 0:
            # Positive torques - different shades of blue
            # Higher torques get brighter blues
            if torque_window >= 5:
                color = '#0000FF'  # Bright blue for 5+ Nm
            elif torque_window == 4:
                color = '#1E90FF'  # Dodger blue for 4-5 Nm
            elif torque_window == 3:
                color = '#4169E1'  # Royal blue for 3-4 Nm
            elif torque_window == 2:
                color = '#6495ED'  # Cornflower blue for 2-3 Nm
            elif torque_window == 1:
                color = '#87CEEB'  # Sky blue for 1-2 Nm
            else:  # 0-1 Nm
                color = '#B0E0E6'  # Powder blue for 0-1 Nm
        else:
            # Negative torques - different shades of red
            # Higher magnitude torques get brighter reds
            if torque_window <= -5:
                color = '#FF0000'  # Bright red for -5 Nm and below
            elif torque_window == -5:
                color = '#FF0000'  # Bright red for -5 to -4 Nm
            elif torque_window == -4:
                color = '#FF1493'  # Deep pink for -4 to -3 Nm
            elif torque_window == -3:
                color = '#FF6347'  # Tomato for -3 to -2 Nm
            elif torque_window == -2:
                color = '#FF7F50'  # Coral for -2 to -1 Nm
            else:  # -1 to 0 Nm
                color = '#FFA07A'  # Light salmon for -1 to 0 Nm
        
        colors.append(color)
    
    return colors

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

# Print torque statistics for both datasets
print(f"\nTorque statistics:")
print(f"LEFT dataset - Total torque range: {df_left['Total_Torque'].min():.2f} to {df_left['Total_Torque'].max():.2f} Nm")
print(f"LEFT dataset - Mean torque: {df_left['Total_Torque'].mean():.2f} Nm")
print(f"RIGHT dataset - Total torque range: {df_right['Total_Torque'].min():.2f} to {df_right['Total_Torque'].max():.2f} Nm")
print(f"RIGHT dataset - Mean torque: {df_right['Total_Torque'].mean():.2f} Nm")

# Create a figure with 8 subplots (4 rows, 2 columns)
fig, axes = plt.subplots(2, 4, figsize=(20, 20))
fig.suptitle('Filtered Data P2 vs P6                           ', 
             fontsize=10, fontweight='bold', y=0.98)

# Plot 1: Flexion vs Rotation
# Left (first dataset)
colors_rot_left = assign_torque_colors(df_rot_left['Total_Torque'])
axes[0, 0].scatter(df_rot_left[column_mapping_left['Rotation']], 
                   df_rot_left[column_mapping_left['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_rot_left)
axes[0, 0].set_xlabel('Internal Rotation (°)             External Rotation (°)', fontsize=8)
axes[0, 0].set_ylabel('Flexion (°)', fontsize=8)
axes[0, 0].set_title('Rotation - P2', fontsize=9, pad=15)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-40, 40)
axes[0, 0].set_ylim(-10, 120)
axes[0, 0].invert_yaxis()

# Right (second dataset)
colors_rot_right = assign_torque_colors(df_rot_right['Total_Torque'])
axes[0, 2].scatter(df_rot_right[column_mapping_right['Rotation']], 
                   df_rot_right[column_mapping_right['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_rot_right)
axes[0, 2].set_xlabel('Internal Rotation (°)             External Rotation (°)', fontsize=8)
axes[0, 2].set_ylabel('Flexion (°)', fontsize=8)
axes[0, 2].set_title('Rotation - P6', fontsize=9, pad=15)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlim(-40, 40)
axes[0, 2].set_ylim(-10, 120)
axes[0, 2].invert_yaxis()

# Plot 2: Flexion vs Adduction
# Left (first dataset)
colors_add_left = assign_torque_colors(df_add_left['Total_Torque'])
axes[0, 1].scatter(df_add_left[column_mapping_left['Adduction']], 
                   df_add_left[column_mapping_left['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_add_left)
axes[0, 1].set_xlabel('Varus (°)      Valgus (°)', fontsize=8)
axes[0, 1].set_ylabel('Flexion (°)', fontsize=8)
axes[0, 1].set_title('Adduction/Abduction - P2', fontsize=9, pad=15)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(-20, 20)
axes[0, 1].set_ylim(-10, 120)
axes[0, 1].invert_yaxis()

# Right (second dataset)
colors_add_right = assign_torque_colors(df_add_right['Total_Torque'])
axes[0, 3].scatter(df_add_right[column_mapping_right['Adduction']], 
                   df_add_right[column_mapping_right['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_add_right)
axes[0, 3].set_xlabel('Varus (°)      Valgus (°)', fontsize=8)
axes[0, 3].set_ylabel('Flexion (°)', fontsize=8)
axes[0, 3].set_title('Adduction/Abduction - P6', fontsize=9, pad=15)
axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].set_xlim(-20, 20)
axes[0, 3].set_ylim(-10, 120)
axes[0, 3].invert_yaxis()

# Plot 3: Flexion vs Anterior_Posterior
# Left (first dataset)
colors_ap_left = assign_torque_colors(df_ap_left['Total_Torque'])
axes[1, 0].scatter(df_ap_left[column_mapping_left['Anterior_Posterior']], 
                   df_ap_left[column_mapping_left['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_ap_left)
axes[1, 0].set_xlabel('Posterior Translation (mm)    Anterior Translation (mm)', fontsize=8)
axes[1, 0].set_ylabel('Flexion (°)', fontsize=8)
axes[1, 0].set_title('Anterior-Posterior - P2', fontsize=9, pad=15)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-20, 20)
axes[1, 0].set_ylim(-10, 120)
axes[1, 0].invert_yaxis()

# Right (second dataset)
colors_ap_right = assign_torque_colors(df_ap_right['Total_Torque'])
axes[1, 2].scatter(df_ap_right[column_mapping_right['Anterior_Posterior']], 
                   df_ap_right[column_mapping_right['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_ap_right)
axes[1, 2].set_xlabel('Posterior Translation (mm)    Anterior Translation (mm)', fontsize=8)
axes[1, 2].set_ylabel('Flexion (°)', fontsize=8)
axes[1, 2].set_title('Anterior-Posterior - P6', fontsize=9, pad=15)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlim(-20, 20)
axes[1, 2].set_ylim(-10, 120)
axes[1, 2].invert_yaxis()

# Plot 4: Flexion vs Medial_Lateral
# Left (first dataset)
colors_ml_left = assign_torque_colors(df_ml_left['Total_Torque'])
axes[1, 1].scatter(df_ml_left[column_mapping_left['Medial_Lateral']], 
                   df_ml_left[column_mapping_left['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_ml_left)
axes[1, 1].set_xlabel('Lateral Translation (mm)     Medial Translation (mm)', fontsize=8)
axes[1, 1].set_ylabel('Flexion (°)', fontsize=8)
axes[1, 1].set_title('Medial-Lateral - P2', fontsize=9, pad=15)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(-50, 50)
axes[1, 1].set_ylim(-10, 120)
axes[1, 1].invert_yaxis()

# Right (second dataset)
colors_ml_right = assign_torque_colors(df_ml_right['Total_Torque'])
axes[1, 3].scatter(df_ml_right[column_mapping_right['Medial_Lateral']], 
                   df_ml_right[column_mapping_right['Flexion']], 
                   alpha=0.6, s=0.5, c=colors_ml_right)
axes[1, 3].set_xlabel('Lateral Translation (mm)     Medial Translation (mm)', fontsize=8)
axes[1, 3].set_ylabel('Flexion (°)', fontsize=8)
axes[1, 3].set_title('Medial-Lateral - P6', fontsize=9, pad=15)
axes[1, 3].grid(True, alpha=0.3)
axes[1, 3].set_xlim(-50, 50)
axes[1, 3].set_ylim(-10, 120)
axes[1, 3].invert_yaxis()

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0.06, 0.9, 0.94])  # Leave space on right
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Create a color legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=8, label='≤ -5 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1493', markersize=8, label='-4 to -3 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6347', markersize=8, label='-3 to -2 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F50', markersize=8, label='-2 to -1 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA07A', markersize=8, label='-1 to 0 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0E0E6', markersize=8, label='0 to 1 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=8, label='1 to 2 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6495ED', markersize=8, label='2 to 3 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=8, label='3 to 4 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF', markersize=8, label='4 to 5 Nm'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=8, label='≥ 5 Nm'),
]

# Add legend to the figure
fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.92, 0.5), 
           ncol=1, fontsize=10, title='Total Torque')


# Show the plots
plt.show()

# Optional: Save the figure
# plt.savefig('combined_flexion_plots_two_datasets.png', dpi=300, bbox_inches='tight')

print("Combined plots with two different datasets generated successfully!")
print(f"LEFT dataset shape: {df_left.shape}")
print(f"RIGHT dataset shape: {df_right.shape}")

# Print summary statistics for both datasets
print("\nSummary statistics for both datasets:")
print(f"\nLEFT dataset (P2):")
print("1. Rotation plot (adduction filtered):")
if len(df_rot_left) > 0:
    print(f"   Flexion range: {df_rot_left[column_mapping_left['Flexion']].min():.1f} to {df_rot_left[column_mapping_left['Flexion']].max():.1f}")
    print(f"   Rotation range: {df_rot_left[column_mapping_left['Rotation']].min():.1f} to {df_rot_left[column_mapping_left['Rotation']].max():.1f}")
    print(f"   Torque range: {df_rot_left['Total_Torque'].min():.2f} to {df_rot_left['Total_Torque'].max():.2f} Nm")

print(f"\nRIGHT dataset (P6):")
print("1. Rotation plot (adduction filtered):")
if len(df_rot_right) > 0:
    print(f"   Flexion range: {df_rot_right[column_mapping_right['Flexion']].min():.1f} to {df_rot_right[column_mapping_right['Flexion']].max():.1f}")
    print(f"   Rotation range: {df_rot_right[column_mapping_right['Rotation']].min():.1f} to {df_rot_right[column_mapping_right['Rotation']].max():.1f}")
    print(f"   Torque range: {df_rot_right['Total_Torque'].min():.2f} to {df_rot_right['Total_Torque'].max():.2f} Nm")

# Print dataset distribution for both datasets
print(f"\nDataset distribution in LEFT data:")
print(df_left['dataset'].value_counts().sort_index())
print(f"\nDataset distribution in RIGHT data:")
print(df_right['dataset'].value_counts().sort_index())