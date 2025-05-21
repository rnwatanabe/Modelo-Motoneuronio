import marimo as mo
import sys
sys.path.append("./../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import os

# set backend to QtAgg (interactive GUI backend)
plt.switch_backend("QtAgg")

# Constants from README
FSAMP = 10240  # Sampling frequency for all datasets

# Data folder path
DATA_PATH = Path("data")

app = mo.App(width="full")

@app.cell
def intro():
    return mo.md("""
    # Motor Unit ISI Analysis
    
    This notebook analyzes Inter-Spike Intervals (ISIs) from motor unit recordings.
    
    It computes:
    - Mean ISI for each motor unit
    - Standard deviation of ISIs for each motor unit
    - Coefficient of variation (CV) of ISIs
    - Mean firing rate
    
    Results are organized by muscle type (VL/VM) and force level.
    """)

@app.cell
def helper_functions():
    def list_mat_files(folder_path, pattern=None):
        """List all .mat files in a given folder, optionally filtering by pattern"""
        mat_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.mat']
        if pattern:
            mat_files = [f for f in mat_files if pattern in f.name]
        return mat_files
    
    def load_ramp_definitions(folder_path):
        """Load ramp definitions from RampDefinition_xx.mat files"""
        ramp_defs = {}
        
        # Find all RampDefinition files
        ramp_files = list_mat_files(folder_path, "RampDefinition")
        
        for file_path in ramp_files:
            # Extract force level from filename (e.g., RampDefinition_05.mat -> 5)
            force_level = int(file_path.stem.split('_')[1])
            
            # Load the file
            try:
                data = sio.loadmat(file_path)
                
                # Check if startMax and stopMax exist
                if 'startMax' in data and 'stopMax' in data:
                    # Initialize dictionary for this force level
                    ramp_defs[force_level] = {}
                    
                    # Get the number of plateaus
                    num_plateaus = len(data['startMax'][0])
                    
                    # Extract start and stop indices for each plateau
                    for i in range(num_plateaus):
                        start_idx = data['startMax'][0, i]
                        stop_idx = data['stopMax'][0, i]
                        ramp_defs[force_level][i] = (start_idx, stop_idx)
                else:
                    print(f"Warning: startMax or stopMax not found in {file_path}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return ramp_defs
    
    def calculate_discharge_differences(pulses):
        """Calculate the differences between consecutive discharges (inter-spike intervals)"""
        # Convert to 1D array if needed
        if pulses.ndim > 1:
            pulses = pulses.flatten()
            
        # Sort pulses to ensure they're in chronological order
        pulses = np.sort(pulses)
        
        # Calculate differences between consecutive pulses
        differences = np.diff(pulses)
        
        # Convert to milliseconds (assuming pulses are in sampling points)
        differences_ms = differences / (FSAMP / 1000)
        
        return differences_ms
    
    return list_mat_files, load_ramp_definitions, calculate_discharge_differences

@app.cell
def folder_selector(list_mat_files, load_ramp_definitions):
    # Find all folders starting with 'P' in the data directory
    p_folders = [f for f in DATA_PATH.iterdir() if f.is_dir() and f.name.startswith('P')]
    
    folder_dropdown = mo.ui.dropdown(
        options=[f.name for f in p_folders],
        value=p_folders[0].name if p_folders else None,
        label="Select Data Folder"
    )
    
    return p_folders, folder_dropdown

@app.cell
def compute_isi_statistics(p_folders, folder_dropdown, list_mat_files, load_ramp_definitions, calculate_discharge_differences):
    if not folder_dropdown:
        return mo.md("No folder selected")
    
    # Get the selected folder
    selected_folder = next((f for f in p_folders if f.name == folder_dropdown), None)
    if not selected_folder:
        return mo.md(f"Folder {folder_dropdown} not found")
    
    # Load ramp definitions
    ramp_defs = load_ramp_definitions(selected_folder)
    if not ramp_defs:
        return mo.md(f"No ramp definitions found in {selected_folder}")
    
    # Function to compute ISI statistics for a specific muscle type
    def compute_isi_stats(muscle_type):
        # List to store statistics for each motor unit
        stats_rows = []
        
        # Get all files for the specified muscle type
        muscle_files = list_mat_files(selected_folder, f"{muscle_type}_")
        
        # Process all files
        for file_path in muscle_files:
            try:
                # Extract force level from filename (e.g., VL_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])
                
                # Check if we have ramp definition for this force level
                if force_level not in ramp_defs:
                    print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                    continue
                
                # Load the data once
                data = sio.loadmat(file_path)
                if 'MUPulses' not in data:
                    print(f"Warning: MUPulses not found in {file_path}")
                    continue
                
                # Get the MUPulses data
                mu_pulses = data['MUPulses']
                n_units = mu_pulses.shape[1]
                
                # Process each plateau
                for plateau_idx in ramp_defs[force_level].keys():
                    # Get plateau start and stop indices
                    plateau_start, plateau_stop = ramp_defs[force_level][plateau_idx]
                    
                    # Process each motor unit
                    for mu in range(n_units):
                        pulses = mu_pulses[0, mu]
                        
                        # Filter pulses to only include those in the plateau phase
                        plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]
                        
                        # Skip if not enough pulses in plateau (need at least 2 for ISI)
                        if len(plateau_pulses) < 2:
                            continue
                        
                        # Calculate ISIs in milliseconds
                        isis = calculate_discharge_differences(plateau_pulses)
                        
                        # Calculate statistics
                        isi_mean = np.mean(isis)
                        isi_std = np.std(isis)
                        isi_cv = isi_std / isi_mean if isi_mean > 0 else np.nan
                        firing_rate = 1000 / isi_mean if isi_mean > 0 else np.nan
                        
                        # Create a row for this motor unit
                        row = {
                            'Folder': selected_folder.name,
                            'File': file_path.name,
                            'Muscle': muscle_type,
                            'Force_Level': force_level,
                            'Plateau': plateau_idx + 1,
                            'MU': mu + 1,
                            'ISI_Mean_ms': isi_mean,
                            'ISI_SD_ms': isi_std,
                            'ISI_CV': isi_cv,
                            'Firing_Rate_Hz': firing_rate,
                            'Num_Spikes': len(plateau_pulses)
                        }
                        stats_rows.append(row)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        return pd.DataFrame(stats_rows)
    
    # Compute ISI statistics for VL and VM
    vl_stats = compute_isi_stats("VL")
    vm_stats = compute_isi_stats("VM")
    
    # Combine statistics
    combined_stats = pd.concat([vl_stats, vm_stats], ignore_index=True)
    
    # Sort by folder, muscle, force level, plateau, and MU
    if not combined_stats.empty:
        combined_stats = combined_stats.sort_values(by=['Folder', 'Muscle', 'Force_Level', 'Plateau', 'MU'])
    
    # Group by muscle and force level
    grouped_stats = combined_stats.groupby(['Muscle', 'Force_Level']).agg({
        'ISI_Mean_ms': 'mean',
        'ISI_SD_ms': 'mean',
        'ISI_CV': 'mean',
        'Firing_Rate_Hz': 'mean',
        'MU': 'count'
    }).reset_index()
    
    grouped_stats = grouped_stats.rename(columns={'MU': 'Num_MUs'})
    
    return combined_stats, grouped_stats

@app.cell
def display_results(compute_isi_statistics):
    if not isinstance(compute_isi_statistics, tuple):
        return compute_isi_statistics
    
    combined_stats, grouped_stats = compute_isi_statistics
    
    if combined_stats.empty:
        return mo.md("No data found for the selected folder")
    
    # Display summary
    summary_text = f"""
    ## ISI Statistics Summary
    
    - Total motor units analyzed: {len(combined_stats)}
    - Muscles: {', '.join(combined_stats['Muscle'].unique())}
    - Force levels: {', '.join(map(str, sorted(combined_stats['Force_Level'].unique())))}
    """
    
    return (
        mo.md(summary_text),
        mo.md("## Grouped Statistics by Muscle and Force Level"),
        mo.ui.table(grouped_stats),
        mo.md("## All Motor Unit Statistics"),
        mo.ui.table(combined_stats)
    )

@app.cell
def plot_cv_vs_rate(compute_isi_statistics):
    if not isinstance(compute_isi_statistics, tuple):
        return compute_isi_statistics
    
    combined_stats, _ = compute_isi_statistics
    
    if combined_stats.empty:
        return mo.md("No data to plot")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with color coding by force level and different markers for different muscles
    for muscle in combined_stats['Muscle'].unique():
        muscle_data = combined_stats[combined_stats['Muscle'] == muscle]
        
        scatter = ax.scatter(
            muscle_data['ISI_CV'], 
            muscle_data['Firing_Rate_Hz'],
            c=muscle_data['Force_Level'],
            marker='o' if muscle == 'VL' else '^',
            s=80, alpha=0.7, edgecolors='k',
            label=muscle
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Force Level (% MVC)')
    
    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title('Relationship Between Firing Rate and Discharge Variability')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return mo.md("## CV vs Firing Rate Plot"), mo.as_html(fig)

@app.cell
def export_to_csv(compute_isi_statistics):
    if not isinstance(compute_isi_statistics, tuple):
        return compute_isi_statistics
    
    combined_stats, grouped_stats = compute_isi_statistics
    
    if combined_stats.empty:
        return mo.md("No data to export")
    
    # Create export button for combined stats
    def export_combined():
        folder_name = combined_stats['Folder'].iloc[0]
        output_file = f"ISI_Statistics_{folder_name}_all_units.csv"
        combined_stats.to_csv(output_file, index=False)
        return f"Exported to {output_file}"
    
    # Create export button for grouped stats
    def export_grouped():
        folder_name = combined_stats['Folder'].iloc[0]
        output_file = f"ISI_Statistics_{folder_name}_grouped.csv"
        grouped_stats.to_csv(output_file, index=False)
        return f"Exported to {output_file}"
    
    export_combined_button = mo.ui.button(export_combined, label="Export All Units Data")
    export_grouped_button = mo.ui.button(export_grouped, label="Export Grouped Data")
    
    return mo.md("## Export Data to CSV"), export_combined_button, export_grouped_button

if __name__ == "__main__":
    app.run()
