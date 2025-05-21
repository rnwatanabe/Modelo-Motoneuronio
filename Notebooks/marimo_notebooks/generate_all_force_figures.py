#!/usr/bin/env python3
"""
Generate Force Figures with CV Information for All P Folders

This script processes all P folders in the data directory and generates
force figures with CV information for all files.
"""

import sys
sys.path.append("./Notebooks/marimo_notebooks")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from pathlib import Path
import os
import p_data_explorer as pde

# Constants
FSAMP = 10240  # Sampling frequency for all datasets

# Data folder path
DATA_PATH = Path("data")

# Output folder for figures
OUTPUT_PATH = Path("Force_Figures")
OUTPUT_PATH.mkdir(exist_ok=True)

def generate_all_force_figures():
    """
    Generate force figures with CV information for all files in the P folders
    """
    print(f"Data path: {DATA_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    # Find all P folders
    p_folders = [f for f in DATA_PATH.iterdir() if f.is_dir() and f.name.startswith("P")]
    
    if not p_folders:
        print("No P folders found in the data directory.")
        return
    
    # Create a results table
    results = []
    
    # Process each P folder
    for p_folder in p_folders:
        print(f"\nProcessing folder: {p_folder.name}")
        
        # Create output folder for this P folder
        p_output_folder = OUTPUT_PATH / p_folder.name
        p_output_folder.mkdir(exist_ok=True)
        
        # Load ramp definitions
        ramp_defs = pde.load_ramp_definitions(p_folder)
        if not ramp_defs:
            print(f"No ramp definitions found in {p_folder.name}, skipping...")
            continue
        
        # Process VL files
        vl_files = [f for f in p_folder.iterdir() if f.suffix.lower() == '.mat' and 'VL_' in f.name]
        for file_path in vl_files:
            try:
                # Extract force level from filename (e.g., VL_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])
                
                # Check if we have ramp definition for this force level
                if force_level not in ramp_defs:
                    print(f"No ramp definition for {force_level}% MVC, skipping {file_path.name}")
                    continue
                
                print(f"  Processing {file_path.name}...")
                
                # Load the data
                data = sio.loadmat(file_path)
                if 'ref_signal' not in data:
                    print(f"No reference signal found in {file_path.name}, skipping...")
                    continue
                
                # Get the reference signal
                ref_signal = data['ref_signal'].flatten()
                
                # Generate force figure with CV information
                fig = pde.plot_force_with_plateaus(data, ramp_defs, force_level, file_path, "VL")
                
                # Save the figure
                output_file = p_output_folder / f"{file_path.stem}_force_plateaus.png"
                fig.savefig(output_file, dpi=300)
                plt.close(fig)
                print(f"    Saved figure to {output_file}")
                
                # Also generate the detailed CV figure
                pde.plot_force_with_plateau_cv(file_path, ref_signal, ramp_defs[force_level], 
                                              force_level, p_output_folder, FSAMP)
                
                # Add to results
                results.append({
                    'Folder': p_folder.name,
                    'File': file_path.name,
                    'Muscle': 'VL',
                    'Force_Level': force_level,
                    'Status': 'Success'
                })
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'Folder': p_folder.name,
                    'File': file_path.name,
                    'Muscle': 'VL',
                    'Force_Level': force_level if 'force_level' in locals() else 'Unknown',
                    'Status': f'Error: {str(e)}'
                })
        
        # Process VM files
        vm_files = [f for f in p_folder.iterdir() if f.suffix.lower() == '.mat' and 'VM_' in f.name]
        for file_path in vm_files:
            try:
                # Extract force level from filename (e.g., VM_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])
                
                # Check if we have ramp definition for this force level
                if force_level not in ramp_defs:
                    print(f"No ramp definition for {force_level}% MVC, skipping {file_path.name}")
                    continue
                
                print(f"  Processing {file_path.name}...")
                
                # Load the data
                data = sio.loadmat(file_path)
                if 'ref_signal' not in data:
                    print(f"No reference signal found in {file_path.name}, skipping...")
                    continue
                
                # Get the reference signal
                ref_signal = data['ref_signal'].flatten()
                
                # Generate force figure with CV information
                fig = pde.plot_force_with_plateaus(data, ramp_defs, force_level, file_path, "VM")
                
                # Save the figure
                output_file = p_output_folder / f"{file_path.stem}_force_plateaus.png"
                fig.savefig(output_file, dpi=300)
                plt.close(fig)
                print(f"    Saved figure to {output_file}")
                
                # Also generate the detailed CV figure
                pde.plot_force_with_plateau_cv(file_path, ref_signal, ramp_defs[force_level], 
                                              force_level, p_output_folder, FSAMP)
                
                # Add to results
                results.append({
                    'Folder': p_folder.name,
                    'File': file_path.name,
                    'Muscle': 'VM',
                    'Force_Level': force_level,
                    'Status': 'Success'
                })
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'Folder': p_folder.name,
                    'File': file_path.name,
                    'Muscle': 'VM',
                    'Force_Level': force_level if 'force_level' in locals() else 'Unknown',
                    'Status': f'Error: {str(e)}'
                })
    
    # Create a DataFrame with the results
    df_results = pd.DataFrame(results)
    
    # Save results to CSV
    results_file = OUTPUT_PATH / "force_figures_generation_results.csv"
    df_results.to_csv(results_file, index=False)
    
    # Print summary
    success_count = len(df_results[df_results['Status'] == 'Success'])
    error_count = len(df_results) - success_count
    
    print("\nGeneration Summary")
    print(f"Total files processed: {len(df_results)}")
    print(f"Successfully generated: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {results_file}")
    print(f"Figures saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_all_force_figures()
