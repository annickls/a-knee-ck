import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Current working directory:", os.getcwd())
#os.chdir(r'C:\files_Annick\Studium Unterlagen\Master\masterarbeit\auswertung\P2 and P6')
print("Changed working directory to:", os.getcwd())

# Read the CSV files for LEFT PLOTS (new dataset)
df1_left = pd.read_csv('20250711_134540_rot_1.csv')
df2_left = pd.read_csv('20250711_134923_var_1.csv')
df3_left = pd.read_csv('20250711_134115_neutral_1.csv')
df4_left = pd.read_csv('20250711_135531_medial_1.csv')
df5_left = pd.read_csv('20250711_135356_anterior_1.csv')
df6_left = pd.read_csv('20250711_135831_var_2.csv')
df7_left = pd.read_csv('20250711_140033_rot_2.csv')
df8_left = pd.read_csv('20250711_140336_anterior_2.csv')
df9_left = pd.read_csv('20250711_140458_medial_2.csv')
df10_left = pd.read_csv('20250711_135916_neutral_2.csv')
df12_left = pd.read_csv('20250711_141417_var_3.csv')
df13_left = pd.read_csv('20250711_141546_rot_3.csv')
df14_left = pd.read_csv('20250711_141714_anterior_3.csv')
df15_left = pd.read_csv('20250711_141801_medial_3.csv')
20250711_135654_lachmann_1
20250711_140740_lachmann_2
20250711_141845_lachmann_3


# Add dataset identifiers for left plots
df1_left['dataset'] = 'rot_1'
df2_left['dataset'] = 'var_1'
df3_left['dataset'] = 'neutral_1'
df4_left['dataset'] = 'medial_1'
df5_left['dataset'] = 'anterior_1'
df6_left['dataset'] = 'var_2'
df7_left['dataset'] = 'rot_2'
df8_left['dataset'] = 'anterior_2'
df9_left['dataset'] = 'medial_2'
df10_left['dataset'] = 'neutral_2'
df12_left['dataset'] = 'var_3'
df13_left['dataset'] = 'rot_3'
df14_left['dataset'] = 'anterior_3'
df15_left['dataset'] = 'medial_3'


# Combine datasets for LEFT PLOTS
df_left = pd.concat([df1_left, df2_left, df3_left, df4_left, df5_left, df6_left, df7_left, 
                    df8_left, df9_left, df10_left, df12_left, df13_left, df14_left, df15_left], 
                  ignore_index=True)
#df_left = pd.concat([df16_left], 
#                  ignore_index=True)

# Read the CSV files for RIGHT PLOTS (original dataset)
df1_right = pd.read_csv('20250711_151442_rot_1.csv')
df2_right = pd.read_csv('20250711_151235_var_1.csv')
df3_right = pd.read_csv('20250711_151100_neutral_1.csv')
df4_right = pd.read_csv('20250711_152019_medial_1.csv')
df5_right = pd.read_csv('20250711_151849_anterior_1.csv')
df6_right = pd.read_csv('20250711_152504_var_2.csv')
df7_right = pd.read_csv('20250711_152806_rot_2.csv')
df8_right = pd.read_csv('20250711_153141_anterior_2.csv')
df9_right = pd.read_csv('20250711_153307_medial_2.csv')
df10_right = pd.read_csv('20250711_152351_neutral_2.csv')
df11_right = pd.read_csv('20250711_153725_neutral_3.csv')
df12_right = pd.read_csv('20250711_154001_var_3.csv')
df13_right = pd.read_csv('20250711_154045_rot_3.csv')
df14_right = pd.read_csv('20250711_154245_anterior_3.csv')
df15_right = pd.read_csv('20250711_154337_medial_3.csv')
df16_right = pd.read_csv('20250711_154423_lachmann_3.csv')
df17_right = pd.read_csv('20250711_152101_lachmann_1.csv')
df18_right = pd.read_csv('20250711_153342_lachmann_2.csv')

# Add dataset identifiers for right plots
df1_right['dataset'] = 'rot_1'
df2_right['dataset'] = 'var_1'
df3_right['dataset'] = 'neutral_1'
df4_right['dataset'] = 'medial_1'
df5_right['dataset'] = 'anterior_1'
df6_right['dataset'] = 'var_2'
df7_right['dataset'] = 'rot_2'
df8_right['dataset'] = 'anterior_2'
df9_right['dataset'] = 'medial_2'
df10_right['dataset'] = 'neutral_2'
df11_right['dataset'] = 'neutral_3'
df12_right['dataset'] = 'var_3'
df13_right['dataset'] = 'rot_3'
df14_right['dataset'] = 'anterior_3'
df15_right['dataset'] = 'medial_3'
df16_right['dataset'] = 'lachman_3'
df17_right['dataset'] = 'lachman_1'
df18_right['dataset'] = 'lachman_2'

# Combine datasets for RIGHT PLOTS
df_right = pd.concat([df1_right, df2_right, df3_right, df4_right, df5_right, df6_right, df7_right, 
                      df8_right, df9_right, df10_right, df11_right, df12_right, df13_right, df14_right, 
                      df15_right, df16_right, df17_right, df18_right], 
                     ignore_index=True)

# Clean column names by stripping whitespace for both datasets
df_left.columns = df_left.columns.str.strip()
df_right.columns = df_right.columns.str.strip()

# Filter ranges
filter_range_narrow = 3 # For left plots 
filter_range_wide = 3     # For right plots 

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
required_columns = ['Flexion', 'Rotation', 'Adduction', 'Anterior_Posterior', 'Medial_Lateral', 
                    'Tx', 'Ty', 'Tz', 'TibiaPosX', 'TibiaPosY', 'TibiaPosZ',
                    'TibiaQuatW', 'TibiaQuatX', 'TibiaQuatY', 'TibiaQuatZ',
                    'Fx', 'Fy', 'Fz']

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



def quaternion_to_transform_matrix(quaternion, position=None):
        """Convert a quaternion and position to a 4x4 transformation matrix."""
        q = np.array(quaternion)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        
        T = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
            [0, 0, 0, 1]
        ])
        
        if position is not None:
            T[0:3, 3] = position
        
        return T


# Calculate total torque for both datasets
#df_left['Total_Torque'] = df_left[column_mapping_left['Tx']] + df_left[column_mapping_left['Ty']] + df_left[column_mapping_left['Tz']]
#df_right['Total_Torque'] = df_right[column_mapping_right['Tx']] + df_right[column_mapping_right['Ty']] + df_right[column_mapping_right['Tz']]

# Extract tibia position data left
left_array_size = len(df_left[column_mapping_left['TibiaPosX']])
tib_position_left = np.zeros((left_array_size, 3))
tib_position_left[:,0] = df_left[column_mapping_left['TibiaPosX']] 
tib_position_left[:,1] = df_left[column_mapping_left['TibiaPosY']] 
tib_position_left[:,2] = df_left[column_mapping_left['TibiaPosZ']] 

# Extract tibia quaternion data
tib_quaternion_left = np.zeros((left_array_size, 4))
tib_quaternion_left[:,0] = df_left[column_mapping_left['TibiaQuatW']] 
tib_quaternion_left[:,1] = df_left[column_mapping_left['TibiaQuatX']] 
tib_quaternion_left[:,2] = df_left[column_mapping_left['TibiaQuatY']] 
tib_quaternion_left[:,3] = df_left[column_mapping_left['TibiaQuatZ']] 

right_array_size = len(df_right[column_mapping_right['Tx']])

# Extract tibia position data
tib_position_right = np.zeros((right_array_size, 3))
tib_position_right[:,0] = df_right[column_mapping_right['TibiaPosX']] 
tib_position_right[:,1] = df_right[column_mapping_right['TibiaPosY']] 
tib_position_right[:,2] = df_right[column_mapping_right['TibiaPosZ']] 

# Extract tibia quaternion data
tib_quaternion_right = np.zeros((right_array_size, 4))
tib_quaternion_right[:,0] = df_right[column_mapping_right['TibiaQuatW']] 
tib_quaternion_right[:,1] = df_right[column_mapping_right['TibiaQuatX']] 
tib_quaternion_right[:,2] = df_right[column_mapping_right['TibiaQuatY']] 
tib_quaternion_right[:,3] = df_right[column_mapping_right['TibiaQuatZ']] 


fx_left = df_left[column_mapping_left['Fx']]
fy_left = df_left[column_mapping_left['Fy']]
fz_left = df_left[column_mapping_left['Fz']]
tx_left = df_left[column_mapping_left['Tx']]
ty_left = df_left[column_mapping_left['Ty']]
tz_left = df_left[column_mapping_left['Tz']]

fx_right = df_right[column_mapping_right['Fx']]
fy_right = df_right[column_mapping_right['Fy']]
fz_right = df_right[column_mapping_right['Fz']]
tx_right = df_right[column_mapping_right['Tx']]
ty_right = df_right[column_mapping_right['Ty']]
tz_right = df_right[column_mapping_right['Tz']]

# Convert to numpy arrays for easier manipulation
fjx_left = fx_left.values.copy()
fjy_left = fy_left.values.copy()
fjz_left = fz_left.values.copy()
tjx_left = tx_left.values.copy()
tjy_left = ty_left.values.copy()
tjz_left = tz_left.values.copy()

# Same for right dataset
fjx_right = fx_right.values.copy()
fjy_right = fy_right.values.copy()
fjz_right = fz_right.values.copy()
tjx_right = tx_right.values.copy()
tjy_right = ty_right.values.copy()
tjz_right = tz_right.values.copy()


# rotate forces back to sensor joint coosys (for now tibia cosys because of data) right
for i in range(len(df_left[column_mapping_left['Tx']])):
    rotation_tibia_plot = quaternion_to_transform_matrix(tib_quaternion_left[i])[:3,:3]
    forces_plot = np.array([fx_left[i], fy_left[i], fz_left[i]])
    forces_tibia_coord = rotation_tibia_plot.T @ forces_plot
    torques_plot = np.array([tx_left[i], ty_left[i], tz_left[i]])
    torques_tibia_coord = rotation_tibia_plot.T @ torques_plot
    fjx_left[i] = forces_tibia_coord[0]
    fjy_left[i] = forces_tibia_coord[1]
    fjz_left[i] = forces_tibia_coord[2]
    tjx_left[i] = torques_tibia_coord[0]
    tjy_left[i] = torques_tibia_coord[1]
    tjz_left[i] = torques_tibia_coord[2]

# rotate forces back to sensor joint coosys (for now tibia cosys because of data) left
for i in range(len(df_right[column_mapping_right['Tx']])):
    rotation_tibia_plot = quaternion_to_transform_matrix(tib_quaternion_right[i])[:3,:3]
    forces_plot = np.array([fx_right[i], fy_right[i], fz_right[i]])
    forces_tibia_coord = rotation_tibia_plot.T @ forces_plot
    torques_plot = np.array([tx_right[i], ty_right[i], tz_right[i]])
    torques_tibia_coord = rotation_tibia_plot.T @ torques_plot
    fjx_right[i] = forces_tibia_coord[0]
    fjy_right[i] = forces_tibia_coord[1]
    fjz_right[i] = forces_tibia_coord[2]
    tjx_right[i] = torques_tibia_coord[0]
    tjy_right[i] = torques_tibia_coord[1]
    tjz_right[i] = torques_tibia_coord[2]




df_left['Total_Torque']= tjx_left
df_right['Total_Torque'] = tjx_right

df_left['Torque_Adduction']= tjz_left
df_right['Torque_Adduction'] = tjz_right

df_left['Force_Anterior']= -fjz_left
df_right['Force_Anterior'] = -fjz_right

df_left['Force_Medial']= fjy_left
df_right['Force_Medial'] = fjy_right

# Function to assign colors based on torque
def assign_torque_colors(torque_values, torquetype):
    colors = []
    for torque in torque_values:
        # Determine the 1Nm window
        torque_window = int(np.floor(torque))
        if torquetype == 1:
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
                elif torque_window == -4:
                    color = '#FF0000'  # Bright red for -5 to -4 Nm
                elif torque_window == -3:
                    color = '#FF1493'  # Deep pink for -4 to -3 Nm
                elif torque_window == -2:
                    color = '#FF6347'  # Tomato for -3 to -2 Nm
                elif torque_window == -1:
                    color = '#FF7F50'  # Coral for -2 to -1 Nm
                else:  # -1 to 0 Nm
                    color = '#FFA07A'  # Light salmon for -1 to 0 Nm
        else:
            if torque >= 0:
                # Positive torques - different shades of blue
                # Higher torques get brighter blues
                if torque_window >= 20:
                    color = '#0000FF'  # Bright blue for 5+ Nm
                elif torque_window == 16:
                    color = '#1E90FF'  # Dodger blue for 4-5 Nm
                elif torque_window == 12:
                    color = '#4169E1'  # Royal blue for 3-4 Nm
                elif torque_window == 8:
                    color = '#6495ED'  # Cornflower blue for 2-3 Nm
                elif torque_window == 4:
                    color = '#87CEEB'  # Sky blue for 1-2 Nm
                else:  # 0-1 Nm
                    color = '#B0E0E6'  # Powder blue for 0-1 Nm
            else:
                # Negative torques - different shades of red
                # Higher magnitude torques get brighter reds
                if torque_window <= -20:
                    color = '#FF0000'  # Bright red for -5 Nm and below
                elif torque_window == -16:
                    color = '#FF0000'  # Bright red for -5 to -4 Nm
                elif torque_window == -12:
                    color = '#FF1493'  # Deep pink for -4 to -3 Nm
                elif torque_window == -8:
                    color = '#FF6347'  # Tomato for -3 to -2 Nm
                elif torque_window == -4:
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
colors_rot_left = assign_torque_colors(df_rot_left['Total_Torque'], 1)
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
colors_rot_right = assign_torque_colors(df_rot_right['Total_Torque'], 1)
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
colors_add_left = assign_torque_colors(df_add_left['Torque_Adduction'], 1)
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
colors_add_right = assign_torque_colors(df_add_right['Torque_Adduction'], 1)
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
colors_ap_left = assign_torque_colors(df_ap_left['Force_Anterior'], 0)
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
colors_ap_right = assign_torque_colors(df_ap_right['Force_Anterior'], 0)
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
colors_ml_left = assign_torque_colors(df_ml_left['Force_Medial'], 0)
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
colors_ml_right = assign_torque_colors(df_ml_right['Force_Medial'], 0)
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
legend_elements_torque = [
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

# Create a color legend
legend_elements_force = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=8, label='≤ -20 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1493', markersize=8, label='-20 to -16 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6347', markersize=8, label='-16 to -12 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F50', markersize=8, label='-12 to -8 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA07A', markersize=8, label='-8 to 4 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0E0E6', markersize=8, label='0 to 4 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=8, label='4 to 8 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6495ED', markersize=8, label='8 to 12 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=8, label='12 to 16 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF', markersize=8, label='16 to 20 N'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=8, label='≥ 20 N'),
]


# Add legend to the figure
fig.legend(handles=legend_elements_torque, loc='center left', bbox_to_anchor=(0.9, 0.75), 
           ncol=1, fontsize=10, title='Corresponding Torque')

fig.legend(handles=legend_elements_force, loc='center left', bbox_to_anchor=(0.9, 0.25), 
           ncol=1, fontsize=10, title='Corresponding Force')

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