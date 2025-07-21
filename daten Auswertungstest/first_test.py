import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('combined_output.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Print available columns for debugging
print("Available columns in your CSV:")
print(df.columns.tolist())
print("\nFirst few rows of data:")
print(df.head())

# Check if required columns exist and map them correctly
required_columns = ['Flexion', 'Rotation', 'Adduction', 'Anterior_Posterior', 'Medial_Lateral']
column_mapping = {}

for col in required_columns:
    if col in df.columns:
        column_mapping[col] = col
    else:
        # Try to find similar column names (case insensitive)
        similar_cols = [c for c in df.columns if col.lower() in c.lower()]
        if similar_cols:
            column_mapping[col] = similar_cols[0]
            print(f"Using '{similar_cols[0]}' for '{col}'")
        else:
            print(f"Warning: Column '{col}' not found in CSV")

# Check if we have all required columns
missing_cols = [col for col in required_columns if col not in column_mapping]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    print("Please check your CSV file and column names")
    exit()

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Flexion vs Various Parameters', fontsize=16, fontweight='bold')

# Plot 1: Flexion vs Rotation
axes[0, 0].scatter(df[column_mapping['Rotation']], df[column_mapping['Flexion']], alpha=0.6, s=2)
axes[0, 0].set_xlabel('Rotation')
axes[0, 0].set_ylabel('Flexion')
axes[0, 0].set_title('Flexion vs Rotation')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-30, 30) 
axes[0, 0].set_ylim(-10, 120) 
axes[0, 0].invert_yaxis()

# Plot 2: Flexion vs Adduction
axes[0, 1].scatter(df[column_mapping['Adduction']], df[column_mapping['Flexion']], alpha=0.6, s=2)
axes[0, 1].set_xlabel('Adduction')
axes[0, 1].set_ylabel('Flexion')
axes[0, 1].set_title('Flexion vs Adduction')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(-20, 20) 
axes[0, 1].set_ylim(-10, 120) 
axes[0, 1].invert_yaxis()

# Plot 3: Flexion vs Anterior_Posterior
axes[1, 0].scatter(df[column_mapping['Anterior_Posterior']], df[column_mapping['Flexion']], alpha=0.6, s=2)
axes[1, 0].set_xlabel('Anterior_Posterior')
axes[1, 0].set_ylabel('Flexion')
axes[1, 0].set_title('Flexion vs Anterior_Posterior')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-20, 20) 
axes[1, 0].set_ylim(-10, 120) 
axes[1, 0].invert_yaxis()

# Plot 4: Flexion vs Medial_Lateral
axes[1, 1].scatter(df[column_mapping['Medial_Lateral']], df[column_mapping['Flexion']], alpha=0.6, s=2)
axes[1, 1].set_xlabel('Medial_Lateral')
axes[1, 1].set_ylabel('Flexion')
axes[1, 1].set_title('Flexion vs Medial_Lateral')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(-50, 50) 
axes[1, 1].set_ylim(-10, 120) 
axes[1, 1].invert_yaxis()

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()

# Optional: Save the figure
#plt.savefig('flexion_plots.png', dpi=300, bbox_inches='tight')

print("Plots generated successfully!")
print(f"Data shape: {df.shape}")
print(f"Available columns: {list(df.columns)}")