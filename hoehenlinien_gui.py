import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate


def calculate_bin_weights(values, bin_centers, weight_type='gaussian', sigma_factor=0.3):
    """
    Calculate weights for values within bins, giving higher weight to values 
    closer to bin centers.
    """
    weights = np.ones_like(values)
    
    for i, (val, center) in enumerate(zip(values, bin_centers)):
        distance = abs(val - center)
        
        if weight_type == 'gaussian':
            sigma = bin_size * sigma_factor  # bin_size is global variable
            weights[i] = np.exp(-(distance**2) / (2 * sigma**2))
            
        elif weight_type == 'triangular':
            max_distance = bin_size / 2
            weights[i] = max(0, 1 - distance / max_distance)
            
        elif weight_type == 'quadratic':
            max_distance = bin_size / 2
            if distance <= max_distance:
                weights[i] = 1 - (distance / max_distance)**2
            else:
                weights[i] = 0
    
    return weights

def apply_moving_average(data, window_size=3, method='simple'):
    """
    Apply moving average to smooth the data.
    
    Parameters:
    - data: pandas DataFrame with columns to smooth
    - window_size: size of the moving window (must be odd for centered window)
    - method: 'simple', 'weighted', or 'exponential'
    
    Returns:
    - smoothed DataFrame
    """
    if len(data) < window_size:
        return data.copy()
    
    # Ensure window size is odd for centered window
    if window_size % 2 == 0:
        window_size += 1
    
    smoothed_data = data.copy()
    
    if method == 'simple':
        # Simple moving average
        smoothed_data['rotation'] = data['rotation'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        smoothed_data['flexion'] = data['flexion'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        smoothed_data['tjx'] = data['tjx'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        
    elif method == 'weighted':
        # Weighted moving average (more weight to center points)
        def weighted_average(series, window_size):
            # Create triangular weights (more weight in center)
            weights = np.bartlett(window_size)
            weights = weights / weights.sum()
            
            result = series.copy()
            half_window = window_size // 2
            
            for i in range(len(series)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(series), i + half_window + 1)
                window_data = series.iloc[start_idx:end_idx]
                
                # Adjust weights for edge cases
                if len(window_data) < window_size:
                    if i < half_window:
                        # Near beginning
                        used_weights = weights[-(len(window_data)):]
                    else:
                        # Near end
                        used_weights = weights[:len(window_data)]
                    used_weights = used_weights / used_weights.sum()
                else:
                    used_weights = weights
                
                result.iloc[i] = np.average(window_data, weights=used_weights)
            
            return result
        
        smoothed_data['rotation'] = weighted_average(data['rotation'], window_size)
        smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
        smoothed_data['tjx'] = weighted_average(data['tjx'], window_size)
        
    elif method == 'exponential':
        # Exponential moving average
        alpha = 2.0 / (window_size + 1)
        smoothed_data['rotation'] = data['rotation'].ewm(alpha=alpha, adjust=False).mean()
        smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
        smoothed_data['tjx'] = data['tjx'].ewm(alpha=alpha, adjust=False).mean()
    
    return smoothed_data

# Read the data
#file_path = '/home/annick/a-knee-ck/auswertung/auswertung_hoehenlinien/20250610_180026_0deg_neutral.csv'
file_path = r'C:\files_Annick\Studium Unterlagen\Master\masterarbeit\a-knee-ck\auswertung\auswertung_hoehenlinien\20250610_180026_0deg_neutral.csv'

try:
    df = pd.read_csv(file_path, comment='#')
    print(f"Successfully loaded data from: {file_path}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please check the file path and make sure the file exists.")
    exit(1)

# Extract the relevant columns
tx = df.iloc[:, 6]  # Tx column (5th column, 0-indexed)
ty = df.iloc[:, 7]
tz = df.iloc[:, 8]
fx = df.iloc[:, 3]
fy = df.iloc[:, 4]
fz = df.iloc[:, 5]

tjx = tx + fz * 0.043 - fy * 0.226 #claculation for torque in the knee joint

flexion = df.iloc[:, 21]  # Flexion column (22nd column, 0-indexed)
rotation = df.iloc[:, 23]  # Rotation column (24th column, 0-indexed)


# Debug: Check the data distribution first
print(f"\nData distribution analysis:")
print(f"tjx range: {tjx.min():.4f} to {tjx.max():.4f}")
print(f"Rotation range: {rotation.min():.2f} to {rotation.max():.2f}")
print(f"Flexion range: {flexion.min():.2f} to {flexion.max():.2f}")

# Create bins for tjx (torque)
bin_size = 0.4
tjx_min = tjx.min()
tjx_max = tjx.max()
tjx_bins = np.arange(tjx_min, tjx_max + bin_size, bin_size)

# Create bins for flexion angles (for averaging)
flexion_bin_size = 0.5  # degrees - adjust this for more/less smoothing
flexion_min = flexion.min()
flexion_max = flexion.max()
flexion_bins = np.arange(flexion_min, flexion_max + flexion_bin_size, flexion_bin_size)

# Smoothing parameters - adjust these to control interpolation
INTERPOLATION_KIND = 'linear'  # Options: 'linear', 'quadratic', 'cubic'
SMOOTHING_FACTOR = 2  # Multiplier for interpolation points (1 = no extra points, 2 = double, etc.)
MIN_POINTS_FOR_SMOOTHING = 2  # Minimum points needed before applying interpolation

# Moving average parameters
MOVING_AVERAGE_WINDOW = 13  # Size of moving average window (must be odd)
MOVING_AVERAGE_METHOD = 'weighted'  # Options: 'simple', 'weighted', 'exponential'
APPLY_MOVING_AVERAGE = True  # Set to False to disable moving average

print(f"\nSmoothing parameters:")
print(f"Interpolation kind: {INTERPOLATION_KIND}")
print(f"Smoothing factor: {SMOOTHING_FACTOR}")
print(f"Minimum points for smoothing: {MIN_POINTS_FOR_SMOOTHING}")
print(f"Moving average: {'Enabled' if APPLY_MOVING_AVERAGE else 'Disabled'}")
if APPLY_MOVING_AVERAGE:
    print(f"Moving average window: {MOVING_AVERAGE_WINDOW}")
    print(f"Moving average method: {MOVING_AVERAGE_METHOD}")
print(f"tjx bins: {len(tjx_bins)-1} bins from {tjx_min:.3f} to {tjx_max:.3f}")
print(f"Flexion bins: {len(flexion_bins)-1} bins from {flexion_min:.2f} to {flexion_max:.2f}")

# Assign bin indices
tjx_bin_indices = pd.cut(tjx, tjx_bins, include_lowest=True, labels=False)
flexion_bin_indices = pd.cut(flexion, flexion_bins, include_lowest=True, labels=False)

# Create bin centers
bin_centers = (tjx_bins[:-1] + tjx_bins[1:]) / 2
tjx_bin_centers = bin_centers[tjx_bin_indices.astype(int)]

flexion_bin_centers = (flexion_bins[:-1] + flexion_bins[1:]) / 2
flexion_bin_centers_mapped = flexion_bin_centers[flexion_bin_indices.astype(int)]

# Create DataFrame with bin indices and centers
data_df = pd.DataFrame({
    'tjx_bin': tjx_bin_indices,
    'flexion_bin': flexion_bin_indices,
    'rotation': rotation,
    'flexion': flexion,
    'tjx': tjx,
    'tjx_bin_center': tjx_bin_centers,
    'flexion_bin_center': flexion_bin_centers_mapped
})

# Remove rows with NaN bin indices
data_df = data_df.dropna()

print(f"Valid data points after binning: {len(data_df)}")

# Weighting parameters
WEIGHT_TYPE = 'gaussian'  # Options: 'gaussian', 'triangular', 'quadratic'
SIGMA_FACTOR = 0.2  # Smaller = more concentration at center

print(f"Calculating weighted averages using {WEIGHT_TYPE} weighting...")

# Calculate weighted averages
weighted_groups = []

for (tjx_bin_idx, flexion_bin_idx), group in data_df.groupby(['tjx_bin', 'flexion_bin']):
    if len(group) < 1:
        continue
    
    # Calculate weights based on distance from torque bin center
    tjx_weights = calculate_bin_weights(
        group['tjx'].values, 
        group['tjx_bin_center'].values,
        WEIGHT_TYPE, 
        SIGMA_FACTOR
    )
    
    # Calculate weighted averages
    total_weight = np.sum(tjx_weights)
    if total_weight > 0:
        weighted_rotation = np.sum(group['rotation'].values * tjx_weights) / total_weight
        weighted_flexion = np.sum(group['flexion'].values * tjx_weights) / total_weight
        weighted_tjx = np.sum(group['tjx'].values * tjx_weights) / total_weight
        
        weighted_groups.append({
            'tjx_bin': tjx_bin_idx,
            'flexion_bin': flexion_bin_idx,
            'rotation': weighted_rotation,
            'flexion': weighted_flexion,
            'tjx': weighted_tjx,
            'n_points': len(group),
            'effective_n': total_weight
        })

# Convert to DataFrame
grouped_data = pd.DataFrame(weighted_groups)

print(f"Weighted grouped data points: {len(grouped_data)}")

# Create a colormap
n_tjx_bins = len(tjx_bins) - 1
colors = plt.cm.viridis(np.linspace(0, 1, n_tjx_bins))

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10))

plotted_lines = 0
for tjx_bin_idx in range(n_tjx_bins):
    # Get data for this tjx bin
    bin_data = grouped_data[grouped_data['tjx_bin'] == tjx_bin_idx].copy()
    
    if len(bin_data) < 1:  # Need at least 1 point
        continue
    
    # Sort by flexion for smooth line connection
    bin_data = bin_data.sort_values('flexion')
    
    # Apply moving average if enabled
    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
        print(f"Applying moving average to tjx bin {tjx_bin_idx} ({len(bin_data)} points)...")
        bin_data = apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
    
    # Separate positive, negative, and zero rotation values
    positive_data = bin_data[bin_data['rotation'] > 0].copy()
    negative_data = bin_data[bin_data['rotation'] < 0].copy()
    zero_data = bin_data[bin_data['rotation'] == 0].copy()
    
    # Helper function to plot a subset of data
    def plot_subset(subset_data, subset_name, marker_style='o'):
        if len(subset_data) < 1:
            return 0
        
        # Sort by flexion for smooth line connection
        subset_data = subset_data.sort_values('flexion')
        x_values = subset_data['rotation'].values
        y_values = subset_data['flexion'].values
        
        # Plot line if we have enough points
        if len(subset_data) >= 2:
            # Apply smoothing/interpolation for smoother lines
            if len(subset_data) >= MIN_POINTS_FOR_SMOOTHING:
                try:
                    # Create interpolation points based on smoothing factor
                    if SMOOTHING_FACTOR > 1:
                        y_smooth = np.linspace(y_values.min(), y_values.max(), 
                                             len(y_values) * SMOOTHING_FACTOR)
                    else:
                        y_smooth = y_values
                    
                    # Use interp1d for interpolation
                    f = interpolate.interp1d(y_values, x_values, kind=INTERPOLATION_KIND, 
                                           bounds_error=False, fill_value='extrapolate')
                    x_smooth = f(y_smooth)
                    
                    # Plot the smooth line
                    ax.plot(x_smooth, y_smooth, 
                           color=colors[tjx_bin_idx], 
                           linewidth=2.5, 
                           alpha=0.8)
                    
                except Exception as e:
                    print(f"Smoothing failed for bin {tjx_bin_idx} {subset_name}: {e}")
                    # Fall back to simple line plot
                    ax.plot(x_values, y_values, 
                           color=colors[tjx_bin_idx], 
                           linewidth=2.5, 
                           alpha=0.8)
            else:
                # Simple line plot for bins with few points
                ax.plot(x_values, y_values, 
                       color=colors[tjx_bin_idx], 
                       linewidth=2.5, 
                       alpha=0.8)
        
        # Plot the actual averaged points
        marker_size = 2 if marker_style == 's' else 15
        edge_color = 'black' if marker_style == 's' else 'white'
        edge_width = 0.5 if marker_style == 's' else 0.2
        
        ax.scatter(x_values, y_values, 
                  c=[colors[tjx_bin_idx]], 
                  alpha=0.9, s=marker_size, marker=marker_style,
                  edgecolors=edge_color, linewidth=edge_width, zorder=5)
        
        return len(subset_data)
    
    # Plot each subset separately
    pos_count = plot_subset(positive_data, "positive", 'o')
    neg_count = plot_subset(negative_data, "negative", 'o') 
    zero_count = plot_subset(zero_data, "zero", 's')  # Square markers for zero rotation
    
    total_count = pos_count + neg_count + zero_count
    
    if total_count > 0:
        # Create legend entry
        tjx_range_start = tjx_bins[tjx_bin_idx]
        tjx_range_end = tjx_bins[tjx_bin_idx + 1]
        tjx_middle = (tjx_range_start + tjx_range_end) / 2
        
        # Add moving average info to legend if enabled
        legend_label = f'tjx: {tjx_middle:.3f} (n={total_count}: +{pos_count}/-{neg_count}/0{zero_count})'
        if APPLY_MOVING_AVERAGE:
            legend_label += f' [MA-{MOVING_AVERAGE_METHOD[:3]}]'
        
        # Add a dummy scatter for legend
        ax.scatter([], [], 
                  c=[colors[tjx_bin_idx]], 
                  label=legend_label,
                  s=60)
        
        plotted_lines += 1

print(f"\nPlotted {plotted_lines} contour lines")

# Make the plot symmetrical around x=0
x_range = max(abs(rotation.min()), abs(rotation.max())) * 1.1
ax.set_xlim(-x_range, x_range)

# Add vertical line at x=0 for reference
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Set labels and title
ax.set_xlabel('Rotation (degrees)', fontsize=12)
ax.set_ylabel('Flexion (degrees)', fontsize=12)
title = 'Flexion vs Rotation Contour Lines\n(Averaged by Torque tjx and Flexion Bins'
if APPLY_MOVING_AVERAGE:
    title += f' + {MOVING_AVERAGE_METHOD.title()} Moving Average)'
else:
    title += ')'
ax.set_title(title, fontsize=14)

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
handles, labels = ax.get_legend_handles_labels()
if len(handles) > 15:  # Limit legend entries if too many
    step = max(1, len(handles) // 15)
    handles = handles[::step]
    labels = labels[::step]

ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()

# Final statistics
print(f"\nFinal statistics:")
print(f"Total original data points: {len(df)}")
print(f"Valid data points after binning: {len(data_df)}")
print(f"Averaged data points: {len(grouped_data)}")
print(f"tjx bins with data: {plotted_lines}")
print(f"Average points per tjx bin: {len(grouped_data) / max(1, plotted_lines):.1f}")
if APPLY_MOVING_AVERAGE:
    print(f"Moving average applied: {MOVING_AVERAGE_METHOD} with window size {MOVING_AVERAGE_WINDOW}")

# Show bin distribution
print(f"\nBin distribution:")
bin_counts = grouped_data['tjx_bin'].value_counts().sort_index()
for tjx_bin_idx, count in bin_counts.items():
    tjx_range_start = tjx_bins[int(tjx_bin_idx)]
    tjx_range_end = tjx_bins[int(tjx_bin_idx) + 1]
    print(f"tjx bin {int(tjx_bin_idx)} ({tjx_range_start:.3f}-{tjx_range_end:.3f}): {count} averaged points")