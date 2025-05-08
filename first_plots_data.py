import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from scipy.spatial.transform import Rotation as R
import math

# Read the data correctly using column names from the header
print("Reading data from 20250430_141205_0deg_neutral.txt...")

# Extract column names from the first line (comment line) of the file
with open("20250508_162909_0deg_var.txt", 'r') as f:
    first_line = f.readline().strip()
    
# Remove the # and split by comma
if first_line.startswith('#'):
    column_names = first_line[1:].strip().split(', ')
    data = pd.read_csv("20250508_162909_0deg_var.txt", skiprows=1, names=column_names)
else:
    # If for some reason there's no header, use default pandas read_csv
    data = pd.read_csv("20250508_162909_0deg_var.txt")

# Print the column names to verify
print("Columns in the dataset:", data.columns.tolist())

# Function to calculate angle between two vectors
def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between two vectors"""
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Calculate angle (make sure we're within valid range for arccos)
    cos_angle = min(max(dot_product / (v1_mag * v2_mag), -1.0), 1.0)
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    
    return angle

# Calculate angles between femur and tibia based on their positions
print("Calculating joint angles based on positions...")
angles = []

for i in range(len(data)):
    # Extract position components
    # Assuming columns 7-9 are femur position and 14-16 are tibia position
    femur_pos = np.array([data.iloc[i, 7], data.iloc[i, 8], data.iloc[i, 9]])
    tibia_pos = np.array([data.iloc[i, 14], data.iloc[i, 15], data.iloc[i, 16]])
    
    # For good anatomical representation, we would ideally have:
    # 1. A point for the hip joint
    # 2. A point for the knee joint
    # 3. A point for the ankle joint
    # Then calculate femur vector (hip to knee) and tibia vector (knee to ankle)
    
    # If we don't have all three points, we can use the femur and tibia position vectors
    # as an approximation to calculate the angle between the segments
    
    # Create vectors from origin to each position
    femur_vector = femur_pos
    tibia_vector = tibia_pos
    
    # Calculate angle between vectors
    angle = angle_between_vectors(femur_vector, tibia_vector)
    angles.append(angle)

# Add the calculated angle to the dataframe
data['FemurTibiaAngle'] = angles

# Calculate torque and force magnitudes
data['TorqueMagnitude'] = np.sqrt(data.iloc[:, 4]**2 + data.iloc[:, 5]**2 + data.iloc[:, 6]**2)
data['ForceMagnitude'] = np.sqrt(data.iloc[:, 1]**2 + data.iloc[:, 2]**2 + data.iloc[:, 3]**2)

print(f"Angle range: {min(angles):.2f}° to {max(angles):.2f}°")
print(f"Torque magnitude range: {data['TorqueMagnitude'].min():.2f} to {data['TorqueMagnitude'].max():.2f} N·m")
print(f"Force magnitude range: {data['ForceMagnitude'].min():.2f} to {data['ForceMagnitude'].max():.2f} N")

# Setup plotting with a standard style
try:
    # Try to use seaborn style if available
    plt.style.use('seaborn-darkgrid')  # More widely available version
except:
    # Fall back to default style if seaborn not available
    print("Default matplotlib style will be used.")
    
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create figure with 3 subplots (1 row, 3 columns)
fig = plt.figure(figsize=(18, 6))

# PLOT 1: Angle vs Torque Components
ax1 = fig.add_subplot(131)

# Convert pandas Series to numpy arrays before plotting to avoid compatibility issues
angle_array = data['Timestamp'].to_numpy()
tx_array = data.iloc[:, 4].to_numpy()
ty_array = data.iloc[:, 5].to_numpy()
tz_array = data.iloc[:, 6].to_numpy()
torque_mag_array = data['TorqueMagnitude'].to_numpy()

# Plot all torque components and magnitude against angle
ax1.plot(angle_array, tx_array, 'r-', label='Tx', linewidth=2)
ax1.plot(angle_array, ty_array, 'g-', label='Ty', linewidth=2)
ax1.plot(angle_array, tz_array, 'b-', label='Tz', linewidth=2)
ax1.plot(angle_array, torque_mag_array, 'k--', label='Total Magnitude', linewidth=1.5)

ax1.set_xlabel('Angle between Femur and Tibia (degrees)')
ax1.set_ylabel('Torque Components (N·m)')
ax1.set_title('Joint Angle vs. Torque Components')
ax1.legend()
ax1.grid(True)

# PLOT 2: Tibia Position Path in 3D space
ax2 = fig.add_subplot(132, projection='3d')

# Extract tibia position columns (assuming they are columns 14, 15, 16)
# If the position columns are different, adjust these indices
tibia_pos_x = data.iloc[:, 7].to_numpy()  # Tibia position X
tibia_pos_y = data.iloc[:, 8].to_numpy()  # Tibia position Y
tibia_pos_z = data.iloc[:, 9].to_numpy()  # Tibia position Z
time_array = data.iloc[:, 0].to_numpy()

# Plot the position path
scatter = ax2.scatter(tibia_pos_x, tibia_pos_y, tibia_pos_z, 
                     c=time_array, cmap='viridis', s=40, alpha=0.8)
ax2.plot(tibia_pos_x, tibia_pos_y, tibia_pos_z, 'b-', linewidth=1.5)

ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.set_zlabel('Z position')
ax2.set_title('Tibia Position Path')

# Add colorbar for time reference inside the plot
cbar = fig.colorbar(scatter, ax=ax2, pad=0.1)  # Adjust pad to move colorbar to the side
cbar.set_label('Time (s)')

# PLOT 3: 3D Force Vectors
ax3 = fig.add_subplot(133, projection='3d')

# Create normalized force magnitudes for color mapping
norm = colors.Normalize(vmin=data['ForceMagnitude'].min(), vmax=data['ForceMagnitude'].max())
cmap = plt.cm.plasma

# Scale factor for visualization (adjust as needed)
scale = 0.01

# Draw force vectors as arrows
for i in range(len(data)):
    # Start at origin
    x, y, z = 0, 0, 0
    
    # Force components (columns 1, 2, 3 are Fx, Fy, Fz)
    u, v, w = data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3]
    
    # Color based on magnitude
    magnitude = data.iloc[i]['ForceMagnitude']
    color = cmap(norm(magnitude))
    
    # Plot arrow
    ax3.quiver(x, y, z, u*scale, v*scale, w*scale, color=color, arrow_length_ratio=0.1)

# Set equal aspect ratio for better visualization
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('X Force (N)')
ax3.set_ylabel('Y Force (N)')
ax3.set_zlabel('Z Force (N)')
ax3.set_title('Force Vectors')

# Add colorbar for force magnitude inside the plot
cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, pad=0.1)
cbar2.set_label('Force Magnitude (N)')

# Improve layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)  # Adjust spacing between plots

# Save figure with high resolution
plt.savefig('biomechanical_analysis.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Visualizations created.")
plt.show()
