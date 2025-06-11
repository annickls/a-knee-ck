import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate

# Read the data
file_path = '/home/annick/a-knee-ck/auswertung/auswertung_hoehenlinien/20250610_180026_0deg_neutral.csv'

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
rotation = df.iloc[:, 23]  # Rotation column (24th column, 0-indexed)
flexion = df.iloc[:, 21]  # Flexion column (22nd column, 0-indexed)

# Debug: Check the data distribution first
print(f"\nData distribution analysis:")
print(f"Tx range: {tx.min():.4f} to {tx.max():.4f}")
print(f"Rotation range: {rotation.min():.2f} to {rotation.max():.2f}")
print(f"Flexion range: {flexion.min():.2f} to {flexion.max():.2f}")

# Create bins for Tx (torque)
bin_size = 0.4
tx_min = tx.min()
tx_max = tx.max()
tx_bins = np.arange(tx_min, tx_max + bin_size, bin_size)

# Create bins for flexion angles (for averaging)
flexion_bin_size = 2.0  # degrees - adjust this for more/less smoothing
flexion_min = flexion.min()
flexion_max = flexion.max()
flexion_bins = np.arange(flexion_min, flexion_max + flexion_bin_size, flexion_bin_size)

# Smoothing parameters - adjust these to control interpolation
INTERPOLATION_KIND = 'linear'  # Options: 'linear', 'quadratic', 'cubic'
SMOOTHING_FACTOR = 2  # Multiplier for interpolation points (1 = no extra points, 2 = double, etc.)
MIN_POINTS_FOR_SMOOTHING = 4  # Minimum points needed before applying interpolation

print(f"\nSmoothing parameters:")
print(f"Interpolation kind: {INTERPOLATION_KIND}")
print(f"Smoothing factor: {SMOOTHING_FACTOR}")
print(f"Minimum points for smoothing: {MIN_POINTS_FOR_SMOOTHING}")
print(f"Tx bins: {len(tx_bins)-1} bins from {tx_min:.3f} to {tx_max:.3f}")
print(f"Flexion bins: {len(flexion_bins)-1} bins from {flexion_min:.2f} to {flexion_max:.2f}")

# Assign bin indices
tx_bin_indices = pd.cut(tx, tx_bins, include_lowest=True, labels=False)
flexion_bin_indices = pd.cut(flexion, flexion_bins, include_lowest=True, labels=False)

# Create DataFrame with bin indices
data_df = pd.DataFrame({
    'tx_bin': tx_bin_indices,
    'flexion_bin': flexion_bin_indices,
    'rotation': rotation,
    'flexion': flexion,
    'tx': tx
})

# Remove rows with NaN bin indices
data_df = data_df.dropna()

print(f"Valid data points after binning: {len(data_df)}")

# Group by tx_bin and flexion_bin, then calculate mean rotation for each group
grouped_data = data_df.groupby(['tx_bin', 'flexion_bin']).agg({
    'rotation': 'mean',
    'flexion': 'mean',  # This will be approximately the center of the flexion bin
    'tx': 'mean'
}).reset_index()

print(f"Grouped data points: {len(grouped_data)}")

# Create a colormap
n_tx_bins = len(tx_bins) - 1
colors = plt.cm.viridis(np.linspace(0, 1, n_tx_bins))

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10))

plotted_lines = 0
for tx_bin_idx in range(n_tx_bins):
    # Get data for this tx bin
    bin_data = grouped_data[grouped_data['tx_bin'] == tx_bin_idx].copy()
    
    if len(bin_data) < 1:  # Need at least 1 point
        continue
    
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
                           color=colors[tx_bin_idx], 
                           linewidth=2.5, 
                           alpha=0.8)
                    
                except Exception as e:
                    print(f"Smoothing failed for bin {tx_bin_idx} {subset_name}: {e}")
                    # Fall back to simple line plot
                    ax.plot(x_values, y_values, 
                           color=colors[tx_bin_idx], 
                           linewidth=2.5, 
                           alpha=0.8)
            else:
                # Simple line plot for bins with few points
                ax.plot(x_values, y_values, 
                       color=colors[tx_bin_idx], 
                       linewidth=2.5, 
                       alpha=0.8)
        
        # Plot the actual averaged points
        marker_size = 50 if marker_style == 's' else 40
        edge_color = 'black' if marker_style == 's' else 'white'
        edge_width = 1.5 if marker_style == 's' else 1
        
        ax.scatter(x_values, y_values, 
                  c=[colors[tx_bin_idx]], 
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
        tx_range_start = tx_bins[tx_bin_idx]
        tx_range_end = tx_bins[tx_bin_idx + 1]
        tx_middle = (tx_range_start + tx_range_end) / 2
        
        # Add a dummy scatter for legend
        ax.scatter([], [], 
                  c=[colors[tx_bin_idx]], 
                  label=f'Tx: {tx_middle:.3f} (n={total_count}: +{pos_count}/-{neg_count}/0{zero_count})',
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
ax.set_title('Flexion vs Rotation Contour Lines\n(Averaged by Torque Tx and Flexion Bins)', fontsize=14)

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
print(f"Tx bins with data: {plotted_lines}")
print(f"Average points per Tx bin: {len(grouped_data) / max(1, plotted_lines):.1f}")

# Show bin distribution
print(f"\nBin distribution:")
bin_counts = grouped_data['tx_bin'].value_counts().sort_index()
for tx_bin_idx, count in bin_counts.items():
    tx_range_start = tx_bins[int(tx_bin_idx)]
    tx_range_end = tx_bins[int(tx_bin_idx) + 1]
    print(f"Tx bin {int(tx_bin_idx)} ({tx_range_start:.3f}-{tx_range_end:.3f}): {count} averaged points")
