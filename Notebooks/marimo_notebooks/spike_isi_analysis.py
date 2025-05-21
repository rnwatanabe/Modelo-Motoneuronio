import marimo as mo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import scipy.stats as stats
from scipy.optimize import curve_fit
import scipy.io as sio
import csv

# Set matplotlib backend to QtAgg for interactive plots
plt.switch_backend("QtAgg")

__generated_with = "0.13.6"
app = mo.App()

# Constants
FSAMP = 10240  # Sampling frequency for all datasets


@app.cell
def intro():
    return mo.md("""
    # Spike Time Analysis and ISI Computation

    This notebook analyzes CSV files containing spike times from motor units, computes Inter-Spike Intervals (ISIs),
    and analyzes the data taking into consideration the plateau phases.

    It computes:
    - ISIs for each motor unit
    - Mean ISI for each motor unit
    - Standard deviation of ISIs
    - Coefficient of variation (CV) of ISIs
    - Mean firing rate

    The analysis focuses on plateau phases where the force is approximately constant.
    """)


@app.cell
def _(mo):
    # Define the folder selection widget
    csv_folders = [
        p for p in Path('.').glob('*_CSV')
        if p.is_dir() and any(f.suffix.lower() == '.csv' for f in p.glob('*'))
    ]

    # Add TA plateau spikes folder if it exists
    ta_folder = Path('TA_plateau_spikes_csv')
    if ta_folder.exists() and ta_folder.is_dir():
        csv_folders.append(ta_folder)

    # Create folder selector widget
    folder_selector = mo.ui.dropdown(
        options={str(folder): folder for folder in csv_folders},
        value=csv_folders[0] if csv_folders else None,
        label="Select folder with spike CSV files"
    )

    return folder_selector


@app.cell
def _(folder_selector):
    # Get list of CSV files in the selected folder
    selected_folder = folder_selector.value

    if selected_folder is None:
        return mo.md("No folder selected or no CSV folders found.")

    csv_files = list(selected_folder.glob('*.csv'))

    if not csv_files:
        return mo.md(f"No CSV files found in {selected_folder}")

    # Display the list of CSV files
    file_list = "\n".join([f"- {f.name}" for f in csv_files])

    return mo.md(f"""
    ## Files in {selected_folder}

    Found {len(csv_files)} CSV files:

    {file_list}
    """)


@app.cell
def _(np, pd):
    def load_csv_file(file_path):
        """
        Load a CSV file containing spike times
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the required columns exist
            required_cols = ['Spike_Time', 'MU', 'Plateau']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {file_path} is missing required columns: {required_cols}")
                return None

            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_isis(spike_times):
        """
        Calculate Inter-Spike Intervals (ISIs) from spike times
        """
        # Sort spike times to ensure they're in chronological order
        sorted_times = np.sort(spike_times)

        # Calculate differences between consecutive spikes (ISIs)
        isis = np.diff(sorted_times)

        return isis

    def compute_isi_statistics(isis):
        """
        Compute statistics for ISIs
        """
        if len(isis) < 2:
            return {
                'mean': np.nan,
                'std': np.nan,
                'cv': np.nan,
                'firing_rate': np.nan
            }

        mean_isi = np.mean(isis)
        std_isi = np.std(isis)
        cv = std_isi / mean_isi if mean_isi > 0 else np.nan
        firing_rate = 1000 / mean_isi if mean_isi > 0 else np.nan  # Convert to Hz assuming ISIs are in ms

        return {
            'mean': mean_isi,
            'std': std_isi,
            'cv': cv,
            'firing_rate': firing_rate
        }

    return load_csv_file, calculate_isis, compute_isi_statistics


@app.cell
def _(load_csv_file, calculate_isis, compute_isi_statistics, folder_selector, pd, np):
    def analyze_folder(folder_path):
        """
        Analyze all CSV files in a folder
        """
        # Get all CSV files in the folder
        csv_files = list(folder_path.glob('*.csv'))

        if not csv_files:
            return mo.md(f"No CSV files found in {folder_path}")

        # List to store results
        results = []

        # Process each file
        for file_path in csv_files:
            # Load the CSV file
            df = load_csv_file(file_path)
            if df is None:
                continue

            # Extract metadata from filename or file content
            muscle = df['Muscle'].iloc[0] if 'Muscle' in df.columns else 'Unknown'
            force_level = df['Force_Level'].iloc[0] if 'Force_Level' in df.columns else 'Unknown'

            # Get unique motor units and plateaus
            motor_units = df['MU'].unique()
            plateaus = df['Plateau'].unique()

            # Process each motor unit and plateau
            for mu in motor_units:
                for plateau in plateaus:
                    # Filter data for this motor unit and plateau
                    mu_plateau_data = df[(df['MU'] == mu) & (df['Plateau'] == plateau)]

                    # Skip if not enough spikes
                    if len(mu_plateau_data) < 2:
                        continue

                    # Calculate ISIs
                    isis = calculate_isis(mu_plateau_data['Spike_Time'].values)

                    # Compute statistics
                    stats = compute_isi_statistics(isis)

                    # Create a result row
                    row = {
                        'File': file_path.name,
                        'Muscle': muscle,
                        'Force_Level': force_level,
                        'MU': mu,
                        'Plateau': plateau,
                        'Num_Spikes': len(mu_plateau_data),
                        'ISI_Mean_ms': stats['mean'],
                        'ISI_SD_ms': stats['std'],
                        'ISI_CV': stats['cv'],
                        'Firing_Rate_Hz': stats['firing_rate']
                    }
                    results.append(row)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Sort by muscle, force level, plateau, and MU
        if not results_df.empty:
            results_df = results_df.sort_values(by=['Muscle', 'Force_Level', 'Plateau', 'MU'])

        return results_df

    # Analyze the selected folder
    selected_folder = folder_selector.value
    if selected_folder is None:
        return mo.md("No folder selected.")

    results_df = analyze_folder(selected_folder)

    # Group results by muscle and force level
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        grouped_df = results_df.groupby(['Muscle', 'Force_Level']).agg({
            'ISI_Mean_ms': 'mean',
            'ISI_SD_ms': 'mean',
            'ISI_CV': 'mean',
            'Firing_Rate_Hz': 'mean',
            'MU': 'count'
        }).reset_index()

        grouped_df = grouped_df.rename(columns={'MU': 'Num_MUs'})

        return results_df, grouped_df

    return results_df, pd.DataFrame()


@app.cell
def _(mo):
    def display_results(analysis_results):
        """
        Display the results of the analysis
        """
        if isinstance(analysis_results, tuple) and len(analysis_results) == 2:
            results_df, grouped_df = analysis_results

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                # Display grouped results
                grouped_table = mo.ui.table(
                    grouped_df,
                    selection="none",
                    pagination=False,
                    title="Summary by Muscle and Force Level"
                )

                # Display detailed results
                detailed_table = mo.ui.table(
                    results_df,
                    selection="none",
                    pagination=True,
                    page_size=10,
                    title="Detailed Results by Motor Unit and Plateau"
                )

                return mo.vstack([
                    mo.md("## Analysis Results"),
                    grouped_table,
                    mo.md("## Detailed Results"),
                    detailed_table
                ])
            else:
                return mo.md("No results to display.")
        else:
            return mo.md("Invalid analysis results format.")

    return display_results


@app.cell
def _(plt, np, mo):
    def plot_cv_vs_rate(analysis_results):
        """
        Plot coefficient of variation vs mean firing rate with quadratic curve fitting
        """
        if isinstance(analysis_results, tuple) and len(analysis_results) == 2:
            results_df, _ = analysis_results

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))

                # Get unique muscles
                muscles = results_df['Muscle'].unique()

                # Define colors and markers for different muscles
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                markers = ['o', 's', '^', 'D', 'v']

                # Define quadratic function for curve fitting
                def quadratic_func(x, a, b, c):
                    return a * x**2 + b * x + c

                # Plot data for each muscle
                for i, muscle in enumerate(muscles):
                    muscle_data = results_df[results_df['Muscle'] == muscle]

                    # Get x and y data
                    x = muscle_data['Firing_Rate_Hz'].values
                    y = muscle_data['ISI_CV'].values

                    # Remove NaN values
                    valid_idx = ~np.isnan(x) & ~np.isnan(y)
                    x = x[valid_idx]
                    y = y[valid_idx]

                    if len(x) > 3:  # Need at least 3 points for quadratic fit
                        # Plot scatter points
                        ax.scatter(x, y, color=colors[i % len(colors)],
                                  marker=markers[i % len(markers)],
                                  label=f'{muscle} Data')

                        try:
                            # Fit quadratic curve
                            popt, _ = curve_fit(quadratic_func, x, y)

                            # Generate points for smooth curve
                            x_fit = np.linspace(min(x), max(x), 100)
                            y_fit = quadratic_func(x_fit, *popt)

                            # Plot fitted curve
                            ax.plot(x_fit, y_fit, color=colors[i % len(colors)],
                                   linestyle='-', linewidth=2,
                                   label=f'{muscle} Fit: {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.4f}')
                        except:
                            print(f"Could not fit curve for {muscle}")

                # Set labels and title
                ax.set_xlabel('Mean Firing Rate (Hz)')
                ax.set_ylabel('Coefficient of Variation')
                ax.set_title('Coefficient of Variation vs Mean Firing Rate')
                ax.legend()
                ax.grid(True)

                return mo.ui.pyplot(fig)
            else:
                return mo.md("No data to plot.")
        else:
            return mo.md("Invalid analysis results format.")

    return plot_cv_vs_rate


@app.cell
def _(plt, np, mo):
    def plot_spike_raster(analysis_results, file_selector=None):
        """
        Plot spike raster for a selected file
        """
        if isinstance(analysis_results, tuple) and len(analysis_results) == 2:
            results_df, _ = analysis_results

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                # Get unique files
                files = results_df['File'].unique()

                # Create file selector if not provided
                if file_selector is None:
                    file_selector = mo.ui.dropdown(
                        options={f: f for f in files},
                        value=files[0] if len(files) > 0 else None,
                        label="Select file to plot"
                    )

                selected_file = file_selector.value

                if selected_file is None:
                    return mo.md("No file selected."), file_selector

                # Load the original CSV file to get all spike times
                folder = folder_selector.value
                file_path = folder / selected_file

                if not file_path.exists():
                    return mo.md(f"File {selected_file} not found."), file_selector

                # Load the CSV file
                df = load_csv_file(file_path)
                if df is None:
                    return mo.md(f"Could not load {selected_file}."), file_selector

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))

                # Get unique motor units and plateaus
                motor_units = df['MU'].unique()
                plateaus = df['Plateau'].unique()

                # Define colors for different plateaus
                plateau_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

                # Plot spikes for each motor unit and plateau
                for plateau in plateaus:
                    plateau_data = df[df['Plateau'] == plateau]

                    for mu in motor_units:
                        mu_data = plateau_data[plateau_data['MU'] == mu]

                        if len(mu_data) > 0:
                            # Plot spike times
                            ax.scatter(mu_data['Spike_Time'].values,
                                      np.ones_like(mu_data['Spike_Time'].values) * mu,
                                      color=plateau_colors[int(plateau-1) % len(plateau_colors)],
                                      marker='|', s=50)

                # Set labels and title
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Motor Unit')
                ax.set_title(f'Spike Raster Plot - {selected_file}')

                # Add legend for plateaus
                legend_elements = [plt.Line2D([0], [0], marker='|', color=plateau_colors[int(p-1) % len(plateau_colors)],
                                             label=f'Plateau {int(p)}', markersize=10, linestyle='None')
                                  for p in plateaus]
                ax.legend(handles=legend_elements)

                # Set y-ticks to motor unit numbers
                ax.set_yticks(motor_units)
                ax.set_yticklabels([f'MU {int(mu)}' for mu in motor_units])

                # Add grid
                ax.grid(True, axis='x', linestyle='--', alpha=0.7)

                return mo.ui.pyplot(fig), file_selector
            else:
                return mo.md("No data to plot."), file_selector
        else:
            return mo.md("Invalid analysis results format."), file_selector

    return plot_spike_raster


@app.cell
def _(mo):
    def export_to_csv(analysis_results):
        """
        Export analysis results to CSV
        """
        if isinstance(analysis_results, tuple) and len(analysis_results) == 2:
            results_df, grouped_df = analysis_results

            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                # Create export buttons
                def export_detailed():
                    folder_name = folder_selector.value.name
                    output_file = f"ISI_Analysis_{folder_name}_detailed.csv"
                    results_df.to_csv(output_file, index=False)
                    return f"Exported to {output_file}"

                def export_summary():
                    folder_name = folder_selector.value.name
                    output_file = f"ISI_Analysis_{folder_name}_summary.csv"
                    grouped_df.to_csv(output_file, index=False)
                    return f"Exported to {output_file}"

                detailed_button = mo.ui.button(
                    export_detailed,
                    label="Export Detailed Results",
                    kind="primary"
                )

                summary_button = mo.ui.button(
                    export_summary,
                    label="Export Summary Results",
                    kind="primary"
                )

                return mo.hstack([detailed_button, summary_button])
            else:
                return mo.md("No data to export.")
        else:
            return mo.md("Invalid analysis results format.")

    return export_to_csv


@app.cell
def _(mo):
    def generate_plateau_spikes_csv(mat_file_path, output_file_path=None, plateau_info=None):
        """
        Generate a CSV file with spike times from plateau phases

        Parameters:
        -----------
        mat_file_path : Path or str
            Path to the .mat file containing spike data
        output_file_path : Path or str, optional
            Path where to save the CSV file. If None, a default name will be used
        plateau_info : dict, optional
            Dictionary with plateau information. If None, will try to extract from file name or content

        Returns:
        --------
        output_path : Path
            Path to the generated CSV file
        """
        try:
            # Convert to Path object if string
            mat_file_path = Path(mat_file_path)

            # Load the .mat file
            data = sio.loadmat(mat_file_path)

            # Check if MUPulses exists in the data
            if 'MUPulses' not in data:
                print(f"MUPulses not found in {mat_file_path}")
                return None

            # Get the MUPulses data
            mu_pulses = data['MUPulses']
            n_units = mu_pulses.shape[1]

            # Extract muscle and force level from filename or data
            muscle = None
            force_level = None

            # Try to extract from filename
            file_name = mat_file_path.name

            # Check for common patterns in filenames
            if '_VL_' in file_name:
                muscle = 'VL'
            elif '_VM_' in file_name:
                muscle = 'VM'
            elif '_TA_' in file_name:
                muscle = 'TA'

            # Try to extract force level from filename
            import re
            force_match = re.search(r'_(\d+)_', file_name)
            if force_match:
                force_level = int(force_match.group(1))

            # If not found in filename, try to extract from data
            if muscle is None and 'muscle' in data:
                muscle = str(data['muscle'][0])

            if force_level is None and 'force_level' in data:
                force_level = int(data['force_level'][0])

            # Default values if still not found
            if muscle is None:
                muscle = "Unknown"

            if force_level is None:
                force_level = 0

            # Determine plateau information
            plateaus = {}

            # If plateau_info is provided, use it
            if plateau_info is not None:
                plateaus = plateau_info
            else:
                # Try to extract from data
                if 'startMax' in data and 'stopMax' in data:
                    num_plateaus = len(data['startMax'][0])
                    for i in range(num_plateaus):
                        start_idx = data['startMax'][0, i]
                        stop_idx = data['stopMax'][0, i]
                        plateaus[i] = (start_idx, stop_idx)
                else:
                    # Try to load from a CSV file with plateau definitions
                    plateau_csv = mat_file_path.parent / 'ta_plateaus.csv'
                    if plateau_csv.exists():
                        plateau_df = pd.read_csv(plateau_csv)
                        # Find the row for this file
                        file_row = plateau_df[plateau_df['filename'] == file_name]
                        if not file_row.empty:
                            # Extract plateau information
                            for i in range(1, 5):  # Assuming up to 4 plateaus
                                if f'plateau{i}_start' in file_row.columns and f'plateau{i}_end' in file_row.columns:
                                    start = file_row[f'plateau{i}_start'].values[0]
                                    end = file_row[f'plateau{i}_end'].values[0]
                                    if not pd.isna(start) and not pd.isna(end):
                                        plateaus[i-1] = (start * FSAMP, end * FSAMP)  # Convert to sample points

                    # If still no plateaus, create a default one covering the entire recording
                    if not plateaus:
                        # Find the maximum pulse time across all motor units
                        max_pulse_time = 0
                        for i in range(n_units):
                            pulses = mu_pulses[0, i]
                            if len(pulses) > 0:
                                max_pulse_time = max(max_pulse_time, np.max(pulses))

                        # Create a single plateau covering the entire recording
                        plateaus[0] = (0, max_pulse_time)

            # Determine output file path
            if output_file_path is None:
                # Create a default name
                output_dir = mat_file_path.parent / f"{muscle}_plateau_spikes_csv"
                output_dir.mkdir(exist_ok=True)
                output_file_path = output_dir / f"{mat_file_path.stem}_all_plateaus_spikes.csv"

            # Convert to Path object if string
            output_file_path = Path(output_file_path)

            # Create a list to store all spike data
            all_spikes = []

            # Process each motor unit
            for mu_idx in range(n_units):
                # Get pulses for this motor unit
                pulses = mu_pulses[0, mu_idx]

                # Process each plateau
                for plateau_idx, (start_time, end_time) in plateaus.items():
                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= start_time) & (pulses <= end_time)]

                    # Convert to seconds
                    plateau_pulses_sec = plateau_pulses / FSAMP

                    # Add to the list
                    for pulse_time in plateau_pulses_sec:
                        all_spikes.append({
                            'Spike_Time': pulse_time,
                            'MU': mu_idx + 1,  # 1-based indexing for motor units
                            'Muscle': muscle,
                            'Force_Level': force_level,
                            'Plateau': plateau_idx + 1  # 1-based indexing for plateaus
                        })

            # Sort by spike time
            all_spikes.sort(key=lambda x: x['Spike_Time'])

            # Write to CSV
            with open(output_file_path, 'w', newline='') as csvfile:
                # Write header comments
                csvfile.write(f"# File: {mat_file_path.name}, Muscle: {muscle}, Force Level: {force_level}% MVC\n")

                # Write plateau information
                for plateau_idx, (start_time, end_time) in plateaus.items():
                    start_sec = start_time / FSAMP
                    end_sec = end_time / FSAMP
                    csvfile.write(f"# Plateau {plateau_idx+1}: {start_sec:.2f}s - {end_sec:.2f}s\n")

                csvfile.write("# Values are spike times in seconds\n")
                csvfile.write("#\n")

                # Write CSV header and data
                fieldnames = ['Spike_Time', 'MU', 'Muscle', 'Force_Level', 'Plateau']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_spikes)

            print(f"Generated CSV file: {output_file_path}")
            return output_file_path

        except Exception as e:
            print(f"Error processing {mat_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    return generate_plateau_spikes_csv


@app.cell
def _(mo, generate_plateau_spikes_csv):
    def process_mat_files_to_csv(folder_path, file_pattern=None):
        """
        Process all .mat files in a folder and generate CSV files with spike times

        Parameters:
        -----------
        folder_path : Path or str
            Path to the folder containing .mat files
        file_pattern : str, optional
            Pattern to filter files (e.g., '*_VL_*.mat')

        Returns:
        --------
        generated_files : list
            List of generated CSV files
        """
        # Convert to Path object if string
        folder_path = Path(folder_path)

        if not folder_path.exists():
            return mo.md(f"Folder {folder_path} does not exist")

        # Get all .mat files in the folder
        if file_pattern:
            mat_files = list(folder_path.glob(file_pattern))
        else:
            mat_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.mat']

        if not mat_files:
            return mo.md(f"No .mat files found in {folder_path}")

        # Process each file
        generated_files = []
        for mat_file in mat_files:
            output_file = generate_plateau_spikes_csv(mat_file)
            if output_file:
                generated_files.append(output_file)

        # Return results
        if generated_files:
            files_list = "\n".join([f"- {f.name}" for f in generated_files])
            return mo.md(f"""
            ## Generated CSV Files

            Successfully generated {len(generated_files)} CSV files:

            {files_list}
            """)
        else:
            return mo.md("No CSV files were generated. Check the console for errors.")

    # Create folder selector
    folders = [p for p in Path('.').iterdir() if p.is_dir()]

    folder_selector = mo.ui.dropdown(
        options={str(folder): folder for folder in folders},
        value=folders[0] if folders else None,
        label="Select folder with .mat files"
    )

    # Create file pattern input
    file_pattern = mo.ui.text(
        value="*.mat",
        label="File pattern (e.g., *_VL_*.mat)"
    )

    # Create process button
    process_button = mo.ui.button(
        lambda: process_mat_files_to_csv(folder_selector.value, file_pattern.value),
        label="Process Files",
        kind="primary"
    )

    return mo.vstack([
        mo.md("## Generate CSV Files with Spike Times from Plateau Phases"),
        mo.md("Select a folder containing .mat files with spike data, and optionally specify a file pattern to filter files."),
        mo.hstack([folder_selector, file_pattern]),
        process_button
    ])


@app.cell
def _(display_results, plot_cv_vs_rate, plot_spike_raster, export_to_csv, folder_selector, load_csv_file):
    # Analyze the selected folder
    analysis_results = _(load_csv_file, calculate_isis, compute_isi_statistics, folder_selector, pd, np)

    # Display the results
    results_display = display_results(analysis_results)

    # Plot CV vs Rate
    cv_rate_plot = plot_cv_vs_rate(analysis_results)

    # Create file selector for spike raster plot
    if isinstance(analysis_results, tuple) and len(analysis_results) == 2:
        results_df, _ = analysis_results
        if isinstance(results_df, pd.DataFrame) and not results_df.empty:
            files = results_df['File'].unique()
            file_selector = mo.ui.dropdown(
                options={f: f for f in files},
                value=files[0] if len(files) > 0 else None,
                label="Select file to plot"
            )
        else:
            file_selector = None
    else:
        file_selector = None

    # Plot spike raster
    spike_plot, _ = plot_spike_raster(analysis_results, file_selector)

    # Export options
    export_options = export_to_csv(analysis_results)

    return mo.vstack([
        results_display,
        mo.md("## Coefficient of Variation vs Mean Firing Rate"),
        cv_rate_plot,
        mo.md("## Spike Raster Plot"),
        file_selector if file_selector is not None else mo.md("No files to select."),
        spike_plot,
        mo.md("## Export Options"),
        export_options
    ])


if __name__ == "__main__":
    app.run()
