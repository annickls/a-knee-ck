import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def calculate_tjx_tjy(df):
    """
    Calculate TJX and TJY values from force and torque measurements.
    TJX = TZ - FY * delta_x + FX * delta_y - TX (for rotation)
    TJY = TX - FZ * delta_y + FY * delta_z (for joint gaps)
    """
    # Define delta values
    delta_x = 0.077
    delta_y = 0.043
    delta_z = 0.226
    
    # Calculate TJX (for rotation analysis)
    df['TJX'] = df['Tz'] - df['Fy'] * delta_x + df['Fx'] * delta_y - df['Tx']
    
    # Calculate TJY (for joint gap analysis)
    df['TJY'] = df['Tx'] - df['Fz'] * delta_y + df['Fy'] * delta_z
    
    return df

def diagnose_knee_data(csv_file_path):
    """
    Diagnostic script to understand the knee measurement data better.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print("=== KNEE DATA DIAGNOSTIC REPORT ===\n")
    
    # Clean column names (remove leading spaces)
    df.columns = df.columns.str.strip()
    
    # Calculate TJX and TJY
    df = calculate_tjx_tjy(df)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Key measurements analysis
    key_columns = ['Flexion', 'Tx', 'TJX', 'TJY', 'Rotation', 'Medial_Joint_Gap', 'Lateral_Joint_Gap']
    
    for col in key_columns:
        if col in df.columns:
            print(f"=== {col} Analysis ===")
            print(f"Range: {df[col].min():.3f} to {df[col].max():.3f}")
            print(f"Mean: {df[col].mean():.3f}")
            print(f"Std: {df[col].std():.3f}")
            print(f"Non-null values: {df[col].count()}/{len(df)}")
            
            # Show unique values if small number
            unique_vals = df[col].nunique()
            if unique_vals <= 20:
                print(f"Unique values: {sorted(df[col].unique())}")
            print()
    
    # Flexion angle distribution
    print("=== FLEXION ANGLE BINS ===")
    flexion_bins = [-15, 15, 45, 75, 105, 135]
    flexion_labels = ['0°', '30°', '60°', '90°', '120°']
    
    df['Flexion_Bin'] = pd.cut(df['Flexion'], 
                              bins=flexion_bins, 
                              labels=flexion_labels,
                              include_lowest=True)
    
    flexion_counts = df['Flexion_Bin'].value_counts().sort_index()
    print("Data points per flexion angle:")
    for angle, count in flexion_counts.items():
        print(f"  {angle}: {count} data points")
    print()
    
    # TJX and TJY analysis
    if 'TJX' in df.columns and 'TJY' in df.columns:
        print("=== TJX (Rotation Torque) ANALYSIS ===")
        print(f"TJX range: {df['TJX'].min():.3f} to {df['TJX'].max():.3f} Nm")
        print(f"TJX mean: {df['TJX'].mean():.3f} Nm")
        print(f"TJX std: {df['TJX'].std():.3f} Nm")
        print()
        
        print("=== TJY (Joint Gap Torque) ANALYSIS ===")
        print(f"TJY range: {df['TJY'].min():.3f} to {df['TJY'].max():.3f} Nm")
        print(f"TJY mean: {df['TJY'].mean():.3f} Nm")
        print(f"TJY std: {df['TJY'].std():.3f} Nm")
        print()
    
    # Torque analysis (original TX for reference)
    if 'Tx' in df.columns:
        print("=== TORQUE (Tx) ANALYSIS ===")
        print(f"Torque range: {df['Tx'].min():.3f} to {df['Tx'].max():.3f} Nm")
        
        # Define torque ranges
        def categorize_torque_range(tx_value):
            if pd.isna(tx_value):
                return 'NaN'
            if 0 <= tx_value <= 0.5:
                return '0-0.5 Nm'
            elif 0.5 < tx_value <= 1.0:
                return '0.5-1.0 Nm'
            elif -0.5 <= tx_value < 0:
                return '-0.5-0 Nm'
            elif -1.0 <= tx_value < -0.5:
                return '-1.0--0.5 Nm'
            else:
                return 'Other'
        
        df['Torque_Range'] = df['Tx'].apply(categorize_torque_range)
        torque_counts = df['Torque_Range'].value_counts()
        print("Data points per torque range:")
        for torque_range, count in torque_counts.items():
            print(f"  {torque_range}: {count} data points")
        print()
        
        # Cross-tabulation of flexion vs torque
        print("=== FLEXION vs TORQUE CROSS-TABULATION ===")
        crosstab = pd.crosstab(df['Flexion_Bin'], df['Torque_Range'], margins=True)
        print(crosstab)
        print()
    
    # Rotation analysis
    if 'Rotation' in df.columns:
        print("=== ROTATION ANALYSIS ===")
        internal_rot = df[df['Rotation'] < 0]['Rotation']
        external_rot = df[df['Rotation'] > 0]['Rotation']
        
        print(f"Internal rotation (negative): {len(internal_rot)} data points")
        if not internal_rot.empty:
            print(f"  Range: {internal_rot.min():.3f} to {internal_rot.max():.3f}°")
            print(f"  Mean: {internal_rot.mean():.3f}°")
        
        print(f"External rotation (positive): {len(external_rot)} data points")
        if not external_rot.empty:
            print(f"  Range: {external_rot.min():.3f} to {external_rot.max():.3f}°")
            print(f"  Mean: {external_rot.mean():.3f}°")
        print()
    
    # Joint gap analysis
    gap_columns = ['Medial_Joint_Gap', 'Lateral_Joint_Gap']
    for col in gap_columns:
        if col in df.columns:
            print(f"=== {col.upper()} ANALYSIS ===")
            print(f"Range: {df[col].min():.3f} to {df[col].max():.3f} mm")
            print(f"Mean: {df[col].mean():.3f} mm")
            print(f"Std: {df[col].std():.3f} mm")
            print()
    
    # Sample data for each flexion angle
    print("=== SAMPLE DATA BY FLEXION ANGLE ===")
    for angle in flexion_labels:
        angle_data = df[df['Flexion_Bin'] == angle]
        if not angle_data.empty:
            print(f"\n{angle} (n={len(angle_data)}):")
            sample_size = min(3, len(angle_data))
            sample_data = angle_data.head(sample_size)
            
            display_cols = ['Flexion', 'Tx', 'TJX', 'TJY', 'Rotation', 'Medial_Joint_Gap', 'Lateral_Joint_Gap']
            available_cols = [col for col in display_cols if col in df.columns]
            
            for _, row in sample_data.iterrows():
                values = [f"{col}: {row[col]:.3f}" for col in available_cols]
                print(f"  {', '.join(values)}")

def get_torque_range_data_tjx_tjy(df, flexion_angle, target_tjx=None, target_tjy=None, torque_tolerance=0.1):
    """
    Get data points within a specific flexion angle and torque range using TJX or TJY.
    
    Parameters:
    df: DataFrame with the knee data
    flexion_angle: Target flexion angle (e.g., '90°')
    target_tjx: Target TJX value for rotation analysis (e.g., 0.5)
    target_tjy: Target TJY value for joint gap analysis (e.g., 0.5)
    torque_tolerance: Range around target torque (e.g., 0.1 means ±0.1 Nm)
    
    Returns:
    DataFrame with filtered data
    """
    # Filter by flexion angle
    angle_data = df[df['Flexion_Bin'] == flexion_angle]
    
    # Filter by torque range based on which torque type is specified
    if target_tjx is not None:
        # Use TJX for rotation-related analysis
        torque_min = target_tjx - torque_tolerance
        torque_max = target_tjx + torque_tolerance
        filtered_data = angle_data[
            (angle_data['TJX'] >= torque_min) & 
            (angle_data['TJX'] <= torque_max)
        ]
    elif target_tjy is not None:
        # Use TJY for joint gap analysis
        torque_min = target_tjy - torque_tolerance
        torque_max = target_tjy + torque_tolerance
        filtered_data = angle_data[
            (angle_data['TJY'] >= torque_min) & 
            (angle_data['TJY'] <= torque_max)
        ]
    else:
        # Return all data for the flexion angle if no torque specified
        filtered_data = angle_data
    
    return filtered_data

def create_torque_range_table_tjx_tjy(csv_file_path, torque_tolerance=0.1):
    """
    Create a table using TJX and TJY torque range-based calculations.
    TJX is used for rotation analysis, TJY is used for joint gap analysis.
    """
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()  # Remove leading spaces
    
    # Calculate TJX and TJY
    df = calculate_tjx_tjy(df)
    
    # Define flexion angle bins
    flexion_bins = [-15, 15, 45, 75, 105, 135]
    flexion_labels = ['0°', '30°', '60°', '90°', '120°']
    
    df['Flexion_Bin'] = pd.cut(df['Flexion'], 
                              bins=flexion_bins, 
                              labels=flexion_labels,
                              include_lowest=True)
    
    # Define target torque values to analyze
    target_torques = [-1.0, -0.5, 0.0, 0.5, 1.0]  # You can modify this list
    
    result_data = []
    
    # Create separate analyses for rotation (TJX) and joint gaps (TJY)
    for angle in flexion_labels:
        for target_torque in target_torques:
            # Analysis using TJX for rotation
            tjx_data = get_torque_range_data_tjx_tjy(df, angle, target_tjx=target_torque, torque_tolerance=torque_tolerance)
            
            if not tjx_data.empty:
                row_tjx = {
                    'Analysis_Type': 'Rotation (TJX)',
                    'Flexion_Angle': angle,
                    'Target_Torque': f"{target_torque:.1f}",
                    'Torque_Range': f"{target_torque-torque_tolerance:.1f} to {target_torque+torque_tolerance:.1f}",
                    'Data_Points': len(tjx_data),
                    'Actual_Torque_Mean': tjx_data['TJX'].mean(),
                    'Actual_Torque_Std': tjx_data['TJX'].std(),
                    'Rotation_Mean': tjx_data['Rotation'].mean(),
                    'Rotation_Std': tjx_data['Rotation'].std(),
                    'Medial_Gap_Mean': tjx_data['Medial_Joint_Gap'].mean(),
                    'Medial_Gap_Std': tjx_data['Medial_Joint_Gap'].std(),
                    'Lateral_Gap_Mean': tjx_data['Lateral_Joint_Gap'].mean(),
                    'Lateral_Gap_Std': tjx_data['Lateral_Joint_Gap'].std()
                }
            else:
                row_tjx = {
                    'Analysis_Type': 'Rotation (TJX)',
                    'Flexion_Angle': angle,
                    'Target_Torque': f"{target_torque:.1f}",
                    'Torque_Range': f"{target_torque-torque_tolerance:.1f} to {target_torque+torque_tolerance:.1f}",
                    'Data_Points': 0,
                    'Actual_Torque_Mean': np.nan,
                    'Actual_Torque_Std': np.nan,
                    'Rotation_Mean': np.nan,
                    'Rotation_Std': np.nan,
                    'Medial_Gap_Mean': np.nan,
                    'Medial_Gap_Std': np.nan,
                    'Lateral_Gap_Mean': np.nan,
                    'Lateral_Gap_Std': np.nan
                }
            
            # Analysis using TJY for joint gaps
            tjy_data = get_torque_range_data_tjx_tjy(df, angle, target_tjy=target_torque, torque_tolerance=torque_tolerance)
            
            if not tjy_data.empty:
                row_tjy = {
                    'Analysis_Type': 'Joint Gaps (TJY)',
                    'Flexion_Angle': angle,
                    'Target_Torque': f"{target_torque:.1f}",
                    'Torque_Range': f"{target_torque-torque_tolerance:.1f} to {target_torque+torque_tolerance:.1f}",
                    'Data_Points': len(tjy_data),
                    'Actual_Torque_Mean': tjy_data['TJY'].mean(),
                    'Actual_Torque_Std': tjy_data['TJY'].std(),
                    'Rotation_Mean': tjy_data['Rotation'].mean(),
                    'Rotation_Std': tjy_data['Rotation'].std(),
                    'Medial_Gap_Mean': tjy_data['Medial_Joint_Gap'].mean(),
                    'Medial_Gap_Std': tjy_data['Medial_Joint_Gap'].std(),
                    'Lateral_Gap_Mean': tjy_data['Lateral_Joint_Gap'].mean(),
                    'Lateral_Gap_Std': tjy_data['Lateral_Joint_Gap'].std()
                }
            else:
                row_tjy = {
                    'Analysis_Type': 'Joint Gaps (TJY)',
                    'Flexion_Angle': angle,
                    'Target_Torque': f"{target_torque:.1f}",
                    'Torque_Range': f"{target_torque-torque_tolerance:.1f} to {target_torque+torque_tolerance:.1f}",
                    'Data_Points': 0,
                    'Actual_Torque_Mean': np.nan,
                    'Actual_Torque_Std': np.nan,
                    'Rotation_Mean': np.nan,
                    'Rotation_Std': np.nan,
                    'Medial_Gap_Mean': np.nan,
                    'Medial_Gap_Std': np.nan,
                    'Lateral_Gap_Mean': np.nan,
                    'Lateral_Gap_Std': np.nan
                }
            
            result_data.append(row_tjx)
            result_data.append(row_tjy)
    
    result_df = pd.DataFrame(result_data)
    return result_df

def create_pivot_summary_tjx_tjy(df_torque_ranges):
    """
    Create separate pivot table summaries for TJX (rotation) and TJY (joint gaps) analyses.
    """
    # Filter out rows with no data points
    df_filtered = df_torque_ranges[df_torque_ranges['Data_Points'] > 0]
    
    pivot_data = {}
    
    # Separate analyses for TJX and TJY
    for analysis_type in ['Rotation (TJX)', 'Joint Gaps (TJY)']:
        df_type = df_filtered[df_filtered['Analysis_Type'] == analysis_type]
        
        if not df_type.empty:
            if analysis_type == 'Rotation (TJX)':
                # For rotation analysis, focus on rotation values
                measurements = [('Rotation_Mean', f'Rotation (°) - {analysis_type}')]
            else:
                # For joint gap analysis, focus on joint gap values
                measurements = [
                    ('Medial_Gap_Mean', f'Medial Joint Gap (mm) - {analysis_type}'),
                    ('Lateral_Gap_Mean', f'Lateral Joint Gap (mm) - {analysis_type}')
                ]
            
            for col, label in measurements:
                pivot = df_type.pivot(index='Flexion_Angle', 
                                     columns='Target_Torque', 
                                     values=col)
                pivot_data[label] = pivot
    
    return pivot_data

def create_table_plots(pivot_summaries, output_dir='.'):
    """
    Create matplotlib table plots for each pivot summary.
    """
    for measurement, pivot_table in pivot_summaries.items():
        if pivot_table.empty:
            continue
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = pivot_table.round(3).fillna('--')
        
        # Create table
        table = ax.table(cellText=table_data.values,
                        rowLabels=table_data.index,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color the header
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color the row labels
        for i in range(1, len(table_data.index) + 1):
            table[(i, -1)].set_facecolor('#E8F5E8')
            table[(i, -1)].set_text_props(weight='bold')
        
        # Add title
        plt.title(f'{measurement}\nTorque Range Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Save the plot
        safe_filename = measurement.replace('(', '').replace(')', '').replace(' - ', '_').replace(' ', '_').replace('°', 'deg')
        output_path = os.path.join(output_dir, f'{safe_filename}_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Table plot saved: {output_path}")
        
        # Show the plot
        plt.show()
        plt.close()

def save_csv_safely(df, filename, output_dir='.'):
    """
    Safely save CSV file with proper error handling and path creation.
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create full path
        full_path = os.path.join(output_dir, filename)
        
        # Save the CSV
        df.to_csv(full_path, index=False)
        
        # Verify the file was created
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print(f"✅ CSV saved successfully: {full_path}")
            print(f"   File size: {file_size} bytes")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
            return full_path
        else:
            print(f"❌ Error: File was not created at {full_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error saving CSV: {str(e)}")
        print(f"   Attempted path: {full_path}")
        print(f"   Current working directory: {os.getcwd()}")
        return None

def main():
    # Use a more flexible path approach
    csv_file_path = "/home/annick/a-knee-ck/auswertung/20250623_094501_0deg_individual.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: CSV file not found at {csv_file_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please update the csv_file_path variable with the correct path.")
        return
    
    print("Running diagnostic analysis...")
    diagnose_knee_data(csv_file_path)
    
    print("\n" + "="*80)
    print("TJX/TJY TORQUE RANGE-BASED ANALYSIS")
    print("="*80)
    print("TJX = TZ - FY * 0.077 + FX * 0.043 - TX (used for rotation analysis)")
    print("TJY = TX - FZ * 0.043 + FY * 0.226 (used for joint gap analysis)")
    print("="*80)
    
    # Create torque range table with ±0.1 Nm tolerance using TJX and TJY
    torque_range_table = create_torque_range_table_tjx_tjy(csv_file_path, torque_tolerance=0.1)
    
    print("\nDetailed TJX/TJY Torque Range Analysis:")
    print("(Each row shows mean values for data within Target_Torque ± 0.1 Nm)")
    print("Rotation analysis uses TJX, Joint gap analysis uses TJY")
    print("-" * 140)
    print(torque_range_table.to_string(index=False, float_format='%.3f'))
    
    # Create pivot summaries
    pivot_summaries = create_pivot_summary_tjx_tjy(torque_range_table)
    
    print("\n" + "="*80)
    print("PIVOT SUMMARIES BY FLEXION ANGLE AND TORQUE (TJX/TJY)")
    print("="*80)
    
    for measurement, pivot_table in pivot_summaries.items():
        print(f"\n{measurement}:")
        print("-" * 60)
        if not pivot_table.empty:
            print(pivot_table.to_string(float_format='%.3f', na_rep='--'))
        else:
            print("No data available")
    
    # Create output directory
    output_dir = 'knee_analysis_output'
    
    # Save results with improved error handling
    csv_path = save_csv_safely(torque_range_table, 'knee_data_tjx_tjy_ranges.csv', output_dir)
    
    if csv_path:
        print(f"\n✅ Analysis complete! Files saved in: {os.path.abspath(output_dir)}")
    
    # Create table plots
    print(f"\n📊 Creating table plots...")
    create_table_plots(pivot_summaries, output_dir)
    
    # Example: Show specific calculation for 90° and 0.5 Nm using both TJX and TJY
    print("\n" + "="*80)
    print("EXAMPLE: 90° Flexion, 0.5 Nm Torque (±0.1 Nm range)")
    print("="*80)
    
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    df = calculate_tjx_tjy(df)
    flexion_bins = [-15, 15, 45, 75, 105, 135]
    flexion_labels = ['0°', '30°', '60°', '90°', '120°']
    df['Flexion_Bin'] = pd.cut(df['Flexion'], bins=flexion_bins, labels=flexion_labels, include_lowest=True)
    
    # Example with TJX for rotation
    print("--- Using TJX for Rotation Analysis ---")
    tjx_data = get_torque_range_data_tjx_tjy(df, '90°', target_tjx=0.5, torque_tolerance=0.1)
    
    if not tjx_data.empty:
        print(f"Found {len(tjx_data)} data points in TJX range 0.4-0.6 Nm at 90° flexion")
        print(f"Actual TJX values: {tjx_data['TJX'].min():.3f} to {tjx_data['TJX'].max():.3f} Nm")
        print(f"Mean rotation: {tjx_data['Rotation'].mean():.3f}°")
        print(f"Mean medial joint gap: {tjx_data['Medial_Joint_Gap'].mean():.3f} mm")
        print(f"Mean lateral joint gap: {tjx_data['Lateral_Joint_Gap'].mean():.3f} mm")
    else:
        print("No data points found in TJX range 0.4-0.6 Nm")
    
    # Example with TJY for joint gaps
    print("\n--- Using TJY for Joint Gap Analysis ---")
    tjy_data = get_torque_range_data_tjx_tjy(df, '90°', target_tjy=0.5, torque_tolerance=0.1)
    
    if not tjy_data.empty:
        print(f"Found {len(tjy_data)} data points in TJY range 0.4-0.6 Nm at 90° flexion")
        print(f"Actual TJY values: {tjy_data['TJY'].min():.3f} to {tjy_data['TJY'].max():.3f} Nm")
        print(f"Mean medial joint gap: {tjy_data['Medial_Joint_Gap'].mean():.3f} mm")
        print(f"Mean lateral joint gap: {tjy_data['Lateral_Joint_Gap'].mean():.3f} mm")
        print(f"Mean rotation: {tjy_data['Rotation'].mean():.3f}°")
    else:
        print("No data points found in TJY range 0.4-0.6 Nm")

if __name__ == "__main__":
    main()