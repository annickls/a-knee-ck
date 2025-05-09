# Function to create a unit sphere mesh
def create_sphere_mesh(radius=1.0, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from scipy.spatial.transform import Rotation as R
import math
from stl import mesh  # Import numpy-stl for STL file handling
import os

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

# Create figure with 4 subplots (2 rows, 2 columns)
fig = plt.figure(figsize=(18, 12))

# PLOT 1: Angle vs Torque Components
ax1 = fig.add_subplot(221)

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
ax2 = fig.add_subplot(222, projection='3d')

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
ax3 = fig.add_subplot(223, projection='3d')

# Create normalized force magnitudes for color mapping
norm = colors.Normalize(vmin=data['ForceMagnitude'].min(), vmax=data['ForceMagnitude'].max())
cmap = plt.cm.plasma_r  # Reversed plasma colormap

# Scale factor for visualization (adjust as needed)
scale = 0.01

# Draw force vectors as arrows
for i in range(len(data)):
    # Start at origin
    x, y, z = 0, 0, 0
    
    # Force components (columns 1, 2, 3 are Fx, Fy, Fz)
    #u, v, w = data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3]

    #Torque components
    u, v, w = data.iloc[i, 4], data.iloc[i, 5], data.iloc[i, 6]
    
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

# Function to load and transform an STL file
def load_stl_for_plot(ax, stl_file_path, scale_factor, translation, rotation_deg, alpha=0.3, face_color=[0.8, 0.8, 0.9]):
    """Load and transform an STL file for plotting
    
    Args:
        ax: Matplotlib 3D axis
        stl_file_path: Path to STL file
        scale_factor: Scale factor for the model
        translation: [x, y, z] translation
        rotation_deg: [x, y, z] rotation in degrees
        alpha: Transparency level
        face_color: RGB tuple for face color
        
    Returns:
        vertices: Transformed vertices
        success: Boolean indicating if loading was successful
    """
    vertices = None
    success = False
    
    try:
        if os.path.exists(stl_file_path):
            print(f"Loading STL file: {stl_file_path}")
            # Load the STL file
            stl_mesh = mesh.Mesh.from_file(stl_file_path)
            
            # Extract vertices for plotting
            vertices = stl_mesh.vectors.copy()  # Create a copy to modify
            
            # Find the center of the STL model
            x_min, x_max = vertices[:,:,0].min(), vertices[:,:,0].max()
            y_min, y_max = vertices[:,:,1].min(), vertices[:,:,1].max()
            z_min, z_max = vertices[:,:,2].min(), vertices[:,:,2].max()
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            
            # Calculate the max dimension for normalization
            max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min)
            
            # 1. First center the model to origin
            vertices[:,:,0] -= center_x
            vertices[:,:,1] -= center_y
            vertices[:,:,2] -= center_z
            
            # 2. Scale the model
            vertices = vertices * (scale_factor / max_dim)
            
            # 3. Apply rotation (converting degrees to radians)
            # Create rotation matrix from Euler angles (degrees)
            rotation = R.from_euler('xyz', [
                np.radians(rotation_deg[0]),  # X rotation
                np.radians(rotation_deg[1]),  # Y rotation
                np.radians(rotation_deg[2])   # Z rotation
            ], degrees=False)
            
            # Apply rotation to each vertex of each face
            for i in range(vertices.shape[0]):  # For each face
                for j in range(vertices.shape[1]):  # For each vertex of the face
                    # Apply rotation to the vertex
                    vertices[i,j,:] = rotation.apply(vertices[i,j,:])
            
            # 4. Apply translation
            vertices[:,:,0] += translation[0]  # X translation
            vertices[:,:,1] += translation[1]  # Y translation
            vertices[:,:,2] += translation[2]  # Z translation
            
            # Plot the STL mesh
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Create a collection of triangles
            triangle_collection = Poly3DCollection(vertices, alpha=alpha)
            
            # Set face colors
            triangle_collection.set_facecolor(face_color)
            triangle_collection.set_edgecolor('gray')
            
            # Add the collection to the plot
            ax.add_collection3d(triangle_collection)
            
            print(f"STL model loaded successfully.")
            print(f"  - Scale factor: {scale_factor}")
            print(f"  - Translation: {translation}")
            print(f"  - Rotation: {rotation_deg} degrees")
            
            success = True
        else:
            print(f"STL file not found: {stl_file_path}")
    except Exception as e:
        print(f"Error loading STL file: {e}")
    
    return vertices, success

# PLOT 4: Sphere with Force Direction Balls and STL Model
ax4 = fig.add_subplot(224, projection='3d')

# === STL MODEL POSITIONING CONTROLS FOR PLOT 4 ===
stl_file_path = "model.stl"  # Change this to your STL file path
stl_scale_factor = 4.9  # < 1.0: smaller model, > 1.0: larger model
stl_translation = [0.0, 2.2, 0.0]  # [x, y, z] offsets
stl_rotation_deg = [0, 180, -90]  # Default: no rotation

# Load the first STL file for plot 4
vertices, stl_loaded = load_stl_for_plot(ax4, stl_file_path, 
                                         stl_scale_factor, 
                                         stl_translation, 
                                         stl_rotation_deg)

# If STL loading failed, create a wireframe sphere instead
if not stl_loaded:
    sphere_x, sphere_y, sphere_z = create_sphere_mesh(radius=1.0, resolution=12)
    ax4.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.5)

# Create a dictionary to track which direction cubes we've already plotted
plotted_directions = {}

# Size of the cubes on the sphere
cube_size = 0.1

# Resolution for discretizing force directions
direction_resolution = 15  # degrees

# Extract all force vectors from the dataset
force_vectors = []
force_magnitudes = []

for i in range(len(data)):
    # Force components (columns 1, 2, 3 are Fx, Fy, Fz)
    fx = data.iloc[i, 1]
    fy = data.iloc[i, 2]
    fz = data.iloc[i, 3]
    force_vec = np.array([fx, fy, fz])
    
    # Normalize the vector to get direction only
    magnitude = np.linalg.norm(force_vec)
    if magnitude > 0:
        normalized_vec = force_vec / magnitude
        force_vectors.append(normalized_vec)
        force_magnitudes.append(magnitude)

# Function to discretize a direction vector to a grid cell
def discretize_direction(direction, resolution):
    # Convert to spherical coordinates
    theta = np.arctan2(direction[1], direction[0])  # azimuth angle
    phi = np.arccos(direction[2])  # polar angle
    
    # Discretize angles
    theta_bin = int((theta + np.pi) / (2 * np.pi / resolution))
    phi_bin = int(phi / (np.pi / resolution))
    
    return (theta_bin, phi_bin)

# Dictionary to store force magnitudes for each discretized direction
direction_magnitudes = {}

# Process all force vectors
for i, (vec, mag) in enumerate(zip(force_vectors, force_magnitudes)):
    # Discretize the direction
    dir_key = discretize_direction(vec, direction_resolution)
    
    # Add magnitude to this direction
    if dir_key in direction_magnitudes:
        direction_magnitudes[dir_key].append(mag)
    else:
        direction_magnitudes[dir_key] = [mag]

# Function to convert from discretized direction back to 3D coordinates
def direction_to_coords(dir_key, resolution, radius=1.0):
    theta_bin, phi_bin = dir_key
    
    # Convert back to continuous angles
    theta = theta_bin * (2 * np.pi / resolution) - np.pi
    phi = phi_bin * (np.pi / resolution)
    
    # Convert to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return x, y, z

# Plot cubes at each discretized direction with color based on average force magnitude
for dir_key, magnitudes in direction_magnitudes.items():
    # Calculate average magnitude
    avg_magnitude = np.mean(magnitudes)
    
    # Get color from colormap
    color = cmap(norm(avg_magnitude))
    
    # Get coordinates on the unit sphere
    x, y, z = direction_to_coords(dir_key, direction_resolution)
    
    # Create a small ball (sphere) at this location
    # Using scatter with a circle marker ('o')
    ax4.scatter([x], [y], [z], color=color, s=100, marker='o', edgecolors=color, alpha=0.9)
    
    # Add a line from origin to the cube to indicate direction
    ax4.plot([0, x], [0, y], [0, z], color='gray', linestyle='--', alpha=0.3)

# Set plot properties
ax4.set_xlabel('X Direction')
ax4.set_ylabel('Y Direction')
ax4.set_zlabel('Z Direction')
ax4.set_title('Force Direction with STL Model')
ax4.set_box_aspect([1, 1, 1])

# Set plot limits with some margin
ax4.set_xlim([-1.5, 1.5])
ax4.set_ylim([-1.5, 1.5])
ax4.set_zlim([-1.5, 1.5])

# Add colorbar for force magnitude
cbar3 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax4, pad=0.1)
cbar3.set_label('Average Force Magnitude (N)')

# PLOT 5: Individual Force Vectors with Two STL Models
ax5 = fig.add_subplot(235, projection='3d')  # Change to 2 rows, 3 columns layout

# === STL MODEL POSITIONING CONTROLS FOR PLOT 5 (FIRST MODEL) ===
# We can use the same model as in plot 4, but with potentially different parameters
stl1_file_path = "model.stl"  # First STL file path (same as plot 4)
stl1_scale_factor = 4.9  # Scale factor for first model
stl1_translation = [0.0, 2.2, 0.0]  # Translation for first model
stl1_rotation_deg = [0, 180, -90]  # Rotation for first model
stl1_face_color = [0.8, 0.8, 0.9]  # Light blue-gray

# === STL MODEL POSITIONING CONTROLS FOR PLOT 5 (SECOND MODEL) ===
stl2_file_path = "second_model.stl"  # Second STL file path
stl2_scale_factor = 4.0  # Scale factor for second model
stl2_translation = [0.2, -1.4, -1.7]  # Translation for second model
stl2_rotation_deg = [180, -60, 90]  # Rotation for second model
stl2_face_color = [0.9, 0.8, 0.8]  # Light red-gray for distinction

# Load the first STL model for plot 5
vertices1, stl1_loaded = load_stl_for_plot(ax5, stl1_file_path, 
                                          stl1_scale_factor, 
                                          stl1_translation, 
                                          stl1_rotation_deg,
                                          alpha=0.3,
                                          face_color=stl1_face_color)

# Load the second STL model for plot 5
vertices2, stl2_loaded = load_stl_for_plot(ax5, stl2_file_path, 
                                          stl2_scale_factor, 
                                          stl2_translation, 
                                          stl2_rotation_deg,
                                          alpha=0.3,
                                          face_color=stl2_face_color)

# If neither STL was loaded, create a wireframe sphere as fallback
if not (stl1_loaded or stl2_loaded):
    sphere_x, sphere_y, sphere_z = create_sphere_mesh(radius=1.0, resolution=12)
    ax5.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.5)

# Set plot limits with larger margin to accommodate longer vectors
lim = 2
ax5.set_xlim([- lim, lim])
ax5.set_ylim([- lim, lim])
ax5.set_zlim([- lim, lim])

# Find maximum magnitude for scaling
max_magnitude = max(force_magnitudes)

# Scale factor for better visualization (adjust based on your data)
magnitude_scale = 2.0 / max_magnitude  # Scale to 2.0 units for maximum magnitude

# Plot EACH individual force vector
for i, (vec, mag) in enumerate(zip(force_vectors, force_magnitudes)):
    # Normalize the direction vector
    normalized_vec = vec / np.linalg.norm(vec)
    
    # Calculate endpoint based on magnitude
    scaled_magnitude = 1 + (mag * magnitude_scale)
    endpoint = normalized_vec * scaled_magnitude
    
    # Get color from colormap
    color = cmap(norm(mag))
    
    # Draw arrow from origin to endpoint
    ax5.quiver(0, 0, 0,
              endpoint[0], endpoint[1], endpoint[2],
              color=color, arrow_length_ratio=0.05,  # Smaller arrowhead
              linewidth=0.2,         # Thinner line
              alpha=0.3)   
    
    # Add a smaller sphere at the endpoint (reduced size for clarity when many points)
    ax5.scatter(endpoint[0], endpoint[1], endpoint[2],
                color=color, s=20, alpha=0.7)

# Set plot properties
ax5.set_xlabel('X Direction')
ax5.set_ylabel('Y Direction')
ax5.set_zlabel('Z Direction')
ax5.set_title('Individual Force Vectors with Two STL Models')
ax5.set_box_aspect([1, 1, 1])

# Add colorbar for force magnitude
cbar5 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax5, pad=0.1)
cbar5.set_label('Force Magnitude (N)')

# Add legend for the two STL models
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=stl1_face_color, markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=stl2_face_color, markersize=10)
]
ax5.legend(custom_lines, ['First STL Model', 'Second STL Model'], loc='upper right')

# Set equal aspect ratio for 3D plots
for ax in [ax2, ax3, ax4, ax5]:
    ax.set_box_aspect([1, 1, 1])

# Update figure layout
fig = plt.gcf()
fig.set_size_inches(24, 12)  # Wider figure to accommodate 3 columns

# Update layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing between plots

# Save figure with high resolution
plt.savefig('biomechanical_analysis_enhanced_5plots.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Enhanced visualizations created with force magnitude plot.")
plt.show()