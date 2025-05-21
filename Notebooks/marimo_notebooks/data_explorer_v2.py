import sys

sys.path.append("./../")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from pathlib import Path
import os
import csv
import re

# set backend to QtAgg for interactive plots
plt.switch_backend("QtAgg")

# Constants from README
FSAMP = 10240  # Sampling frequency for all datasets

# Data folder path
DATA_PATH = Path("data")

def list_data_folders():
    """List all data folders in the data directory"""
    folders = [f for f in DATA_PATH.iterdir() if f.is_dir()]
    return folders

def list_subfolders(folder_path):
    """List all subfolders in a given folder"""
    subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
    return subfolders

def list_mat_files(folder_path):
    """List all .mat files in a given folder"""
    mat_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.mat']
    return mat_files

def load_mat_file(file_path):
    """Load a .mat file and return its contents"""
    try:
        data = sio.loadmat(file_path)
        # Remove metadata keys (those starting with '__')
        data_keys = [key for key in data.keys() if not key.startswith('__')]
        return data, data_keys
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, []

def plot_time_series(data, key, title="Time Series", max_rows=5):
    """Plot time series data"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the data for the selected key
    selected_data = data[key]

    # If data is 1D, plot directly
    if len(selected_data.shape) == 1:
        ax.plot(selected_data)
        ax.set_title(f"{title} - {key}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")

    # If data is 2D, plot first few rows
    elif len(selected_data.shape) == 2:
        num_rows_to_plot = min(max_rows, selected_data.shape[0])
        print(num_rows_to_plot)
        for i in range(num_rows_to_plot):
            print(selected_data[i])
            ax.plot(selected_data[i].squeeze(), label=f"Series {i+1}")
        ax.legend()
        ax.set_title(f"{title} - {key} (First {num_rows_to_plot} series)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")

    return fig

def plot_heatmap(data, key, title="Heatmap"):
    """Plot data as a heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))

    selected_data = data[key]

    # Check if data is 2D
    if len(selected_data.shape) == 2:
        im = ax.imshow(selected_data, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title} - {key}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Series Index")
    else:
        ax.text(0.5, 0.5, "Data must be 2D for heatmap", ha='center', va='center')
        ax.set_title("Cannot create heatmap")

    return fig

def plot_histogram(data, key, title="Histogram", bins=50):
    """Plot histogram of data"""
    fig, ax = plt.subplots(figsize=(10, 6))

    selected_data = data[key]

    ax.hist(selected_data.flatten(), bins=bins)
    ax.set_title(f"{title} - {key}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    return fig

def export_to_csv(data, key, filename):
    """Export data to CSV file"""
    selected_data = data[key]

    # Convert to DataFrame
    if len(selected_data.shape) == 1:
        df = pd.DataFrame(selected_data, columns=["Value"])
    else:
        df = pd.DataFrame(selected_data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")

def explore_data_interactive():
    """Main function to explore data interactively"""
    print("Motoneuron Data Explorer")
    print("========================")
    print("\nThis program allows you to explore the data in the data folder.")

    # List all data folders
    data_folders = list_data_folders()
    if not data_folders:
        print("No data folders found")
        return

    print("\nAvailable data folders:")
    for i, folder in enumerate(data_folders):
        print(f"{i+1}. {folder.name}")

    # Select a data folder
    folder_idx = int(input("Select a folder (number): ")) - 1
    if folder_idx < 0 or folder_idx >= len(data_folders):
        print("Invalid selection")
        return

    selected_folder = data_folders[folder_idx]

    # Check if the selected folder has subfolders
    subfolders = list_subfolders(selected_folder)

    if subfolders:
        print(f"\nSubfolders in {selected_folder.name}:")
        for i, subfolder in enumerate(subfolders):
            print(f"{i+1}. {subfolder.name}")

        # Select a subfolder
        subfolder_idx = int(input("Select a subfolder (number): ")) - 1
        if subfolder_idx < 0 or subfolder_idx >= len(subfolders):
            print("Invalid selection")
            return

        selected_path = subfolders[subfolder_idx]
    else:
        # No subfolders, use the selected folder directly
        selected_path = selected_folder

    # List .mat files in the selected path
    mat_files = list_mat_files(selected_path)
    if not mat_files:
        print(f"No .mat files found in {selected_path}")
        return

    print(f"\nMAT files in {selected_path.name}:")
    for i, mat_file in enumerate(mat_files):
        print(f"{i+1}. {mat_file.name}")

    # Select a .mat file
    file_idx = int(input("Select a file (number): ")) - 1
    if file_idx < 0 or file_idx >= len(mat_files):
        print("Invalid selection")
        return

    selected_file = mat_files[file_idx]

    # Load the selected .mat file
    data, data_keys = load_mat_file(selected_file)
    if data is None:
        return

    print(f"\nVariables in {selected_file.name}:")
    for i, key in enumerate(data_keys):
        print(f"{i+1}. {key}")

    # Select a variable
    key_idx = int(input("Select a variable (number): ")) - 1
    if key_idx < 0 or key_idx >= len(data_keys):
        print("Invalid selection")
        return

    selected_key = data_keys[key_idx]

    # Display information about the selected variable
    selected_data = data[selected_key]
    print(f"\nVariable: {selected_key}")
    print(f"Type: {type(selected_data).__name__}")
    print(f"Shape: {selected_data.shape if hasattr(selected_data, 'shape') else 'N/A'}")

    # Plot options
    print("\nPlot options:")
    print("1. Time Series")
    print("2. Heatmap")
    print("3. Histogram")

    plot_option = int(input("Select a plot type (number): "))

    if plot_option == 1:
        fig = plot_time_series(data, selected_key, title=f"{selected_folder.name} Data")
    elif plot_option == 2:
        fig = plot_heatmap(data, selected_key, title=f"{selected_folder.name} Data")
    elif plot_option == 3:
        fig = plot_histogram(data, selected_key, title=f"{selected_folder.name} Data")
    else:
        print("Invalid selection")
        return

    plt.tight_layout()
    plt.show()

    # Export option
    export_option = input("\nExport data to CSV? (y/n): ")
    if export_option.lower() == 'y':
        filename = input("Enter filename (without extension): ")
        export_to_csv(data, selected_key, f"{filename}.csv")

def load_and_plot_specific_file(folder_name, subfolder_name=None, file_name=None, variable_name=None):
    """
    Load and plot a specific file without interactive prompts

    Parameters:
    -----------
    folder_name : str
        Name of the data folder (e.g., 'FDI', 'TA', 'P01_iEMG_Ramps')
    subfolder_name : str, optional
        Name of the subfolder if applicable
    file_name : str, optional
        Name of the .mat file to load
    variable_name : str, optional
        Name of the variable to plot
    """
    # Construct the path to the folder
    folder_path = DATA_PATH / folder_name

    if not folder_path.exists():
        print(f"Folder {folder_name} not found")
        return

    # If subfolder is specified, update the path
    if subfolder_name:
        folder_path = folder_path / subfolder_name
        if not folder_path.exists():
            print(f"Subfolder {subfolder_name} not found in {folder_name}")
            return

    # If file_name is not specified, list available files
    if not file_name:
        mat_files = list_mat_files(folder_path)
        if not mat_files:
            print(f"No .mat files found in {folder_path}")
            return

        print(f"\nMAT files in {folder_path}:")
        for i, mat_file in enumerate(mat_files):
            print(f"{i+1}. {mat_file.name}")

        # Select a file
        file_idx = int(input("Select a file (number): ")) - 1
        if file_idx < 0 or file_idx >= len(mat_files):
            print("Invalid selection")
            return

        selected_file = mat_files[file_idx]
    else:
        # Use the specified file name
        selected_file = folder_path / file_name
        if not selected_file.exists():
            print(f"File {file_name} not found in {folder_path}")
            return

    # Load the selected file
    data, data_keys = load_mat_file(selected_file)
    if data is None:
        return

    # If variable_name is not specified, list available variables
    if not variable_name:
        print(f"\nVariables in {selected_file.name}:")
        for i, key in enumerate(data_keys):
            print(f"{i+1}. {key}")

        # Select a variable
        key_idx = int(input("Select a variable (number): ")) - 1
        if key_idx < 0 or key_idx >= len(data_keys):
            print("Invalid selection")
            return

        selected_key = data_keys[key_idx]
    else:
        # Use the specified variable name
        if variable_name not in data_keys:
            print(f"Variable {variable_name} not found in {selected_file.name}")
            print(f"Available variables: {', '.join(data_keys)}")
            return

        selected_key = variable_name

    # Display information about the selected variable
    selected_data = data[selected_key]
    print(f"\nVariable: {selected_key}")
    print(f"Type: {type(selected_data).__name__}")
    print(f"Shape: {selected_data.shape if hasattr(selected_data, 'shape') else 'N/A'}")

    # Plot the data
    fig = plot_time_series(data, selected_key, title=f"{folder_name} Data")
    plt.tight_layout()
    plt.show()

    return data, selected_key

def calculate_discharge_differences(pulses):
    """
    Calculate the differences between consecutive discharges (inter-spike intervals)

    Parameters:
    -----------
    pulses : numpy.ndarray
        Array of pulse timestamps

    Returns:
    --------
    differences : numpy.ndarray
        Array of differences between consecutive pulses (in ms)
    """
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

def calculate_instantaneous_rate(pulses, window_size=1000):
    """
    Calculate instantaneous firing rate over time

    Parameters:
    -----------
    pulses : numpy.ndarray
        Array of pulse timestamps
    window_size : int, optional
        Size of the sliding window in ms

    Returns:
    --------
    times : numpy.ndarray
        Time points
    rates : numpy.ndarray
        Instantaneous firing rates at each time point
    """
    # Convert to 1D array if needed
    if pulses.ndim > 1:
        pulses = pulses.flatten()

    # Sort pulses to ensure they're in chronological order
    pulses = np.sort(pulses)

    # Convert to milliseconds
    pulses_ms = pulses / (FSAMP / 1000)

    # Create time vector (1 ms resolution)
    if len(pulses_ms) > 0:
        t_start = 0
        t_end = pulses_ms[-1] + window_size
        times = np.arange(t_start, t_end, 1)

        # Calculate instantaneous rate at each time point
        rates = np.zeros_like(times, dtype=float)

        for i, t in enumerate(times):
            # Count spikes in window [t-window_size/2, t+window_size/2]
            window_start = t - window_size/2
            window_end = t + window_size/2
            spikes_in_window = np.sum((pulses_ms >= window_start) & (pulses_ms < window_end))

            # Calculate rate in Hz (spikes per second)
            rates[i] = spikes_in_window / (window_size / 1000)

        return times, rates
    else:
        return np.array([]), np.array([])

def plot_discharge_differences(file_path, title=None):
    """
    Plot the differences between consecutive discharges for all motor units in a file

    Parameters:
    -----------
    file_path : Path
        Path to the .mat file
    title : str, optional
        Title for the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Load the .mat file
    try:
        data = sio.loadmat(file_path)

        # Check if MUPulses exists in the data
        if 'MUPulses' not in data:
            print(f"MUPulses not found in {file_path}")
            return None

        # Get the MUPulses data
        mu_pulses = data['MUPulses']

        # Check if reference signal exists
        has_ref_signal = 'ref_signal' in data

        # Create figure with 2 or 3 columns based on whether ref_signal exists
        n_cols = 3 if has_ref_signal else 2
        fig, axes = plt.subplots(mu_pulses.shape[1], n_cols, figsize=(5*n_cols, 3*mu_pulses.shape[1]), sharey='row')

        # If there's only one motor unit, make axes iterable
        if mu_pulses.shape[1] == 1:
            axes = np.array([axes]).reshape(1, -1)

        # Plot discharge differences and rates for each motor unit
        for i in range(mu_pulses.shape[1]):
            # Get pulses for this motor unit
            pulses = mu_pulses[0, i]

            # Calculate differences
            differences = calculate_discharge_differences(pulses)

            # Plot histogram of differences
            axes[i, 0].hist(differences, bins=50, alpha=0.7)
            axes[i, 0].set_title(f"Motor Unit {i+1} - ISI Distribution")
            axes[i, 0].set_ylabel("Count")
            axes[i, 0].set_xlabel("Inter-spike Interval (ms)")

            # Calculate and display statistics
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            cv = std_diff / mean_diff  # Coefficient of variation

            # Add text with statistics
            stats_text = f"Mean: {mean_diff:.2f} ms\nStd: {std_diff:.2f} ms\nCV: {cv:.3f}"
            axes[i, 0].text(0.95, 0.95, stats_text, transform=axes[i, 0].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Calculate and plot instantaneous firing rate
            times, rates = calculate_instantaneous_rate(pulses, window_size=1000)
            if len(times) > 0:
                axes[i, 1].plot(times/1000, rates)  # Convert to seconds for x-axis
                axes[i, 1].set_title(f"Motor Unit {i+1} - Firing Rate")
                axes[i, 1].set_xlabel("Time (s)")
                axes[i, 1].set_ylabel("Firing Rate (Hz)")
                axes[i, 1].grid(True, linestyle='--', alpha=0.7)

                # Add mean rate as horizontal line
                mean_rate = 1000 / mean_diff  # Convert from ms to Hz
                axes[i, 1].axhline(y=mean_rate, color='r', linestyle='--', alpha=0.7)
                axes[i, 1].text(0.02, 0.95, f"Mean Rate: {mean_rate:.2f} Hz",
                             transform=axes[i, 1].transAxes, color='r',
                             verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # If reference signal exists, plot it along with firing rate
                if has_ref_signal:
                    ref_signal = data['ref_signal'].flatten()

                    # Create time vector for reference signal
                    ref_time = np.arange(len(ref_signal)) / FSAMP  # Convert to seconds

                    # Plot reference signal
                    ax_ref = axes[i, 2]
                    ax_ref.plot(ref_time, ref_signal, 'k-', label='Force')
                    ax_ref.set_xlabel('Time (s)')
                    ax_ref.set_ylabel('Force (N)')
                    ax_ref.set_title(f'Motor Unit {i+1} - Force vs. Rate')
                    ax_ref.grid(True, linestyle='--', alpha=0.7)

                    # Create second y-axis for firing rate
                    ax_rate = ax_ref.twinx()
                    ax_rate.plot(times/1000, rates, 'b-', label='Firing Rate')
                    ax_rate.set_ylabel('Firing Rate (Hz)', color='b')
                    ax_rate.tick_params(axis='y', labelcolor='b')

                    # Add legend
                    lines1, labels1 = ax_ref.get_legend_handles_labels()
                    lines2, labels2 = ax_rate.get_legend_handles_labels()
                    ax_ref.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Set title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(file_path.name, fontsize=16)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def plot_mu_summary(file_path, title=None):
    """
    Create a summary plot comparing all motor units in a file

    Parameters:
    -----------
    file_path : Path
        Path to the .mat file
    title : str, optional
        Title for the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    try:
        # Load the .mat file
        data = sio.loadmat(file_path)

        # Check if MUPulses exists in the data
        if 'MUPulses' not in data:
            print(f"MUPulses not found in {file_path}")
            return None

        # Get the MUPulses data
        mu_pulses = data['MUPulses']
        n_units = mu_pulses.shape[1]

        # Check if reference signal exists
        has_ref_signal = 'ref_signal' in data

        # Create figure with 3 or 4 subplots based on whether ref_signal exists
        n_rows = 3 if has_ref_signal else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5*n_rows))

        # 1. Plot mean firing rates for all motor units
        mean_rates = []
        cvs = []

        for i in range(n_units):
            # Get pulses for this motor unit
            pulses = mu_pulses[0, i]

            # Calculate differences
            differences = calculate_discharge_differences(pulses)

            # Calculate mean rate and CV
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            cv = std_diff / mean_diff

            # Convert to Hz
            mean_rate = 1000 / mean_diff

            mean_rates.append(mean_rate)
            cvs.append(cv)

        # Plot mean firing rates
        axes[0].bar(range(1, n_units+1), mean_rates, alpha=0.7)
        axes[0].set_xlabel('Motor Unit')
        axes[0].set_ylabel('Mean Firing Rate (Hz)')
        axes[0].set_title('Mean Firing Rates by Motor Unit')
        axes[0].set_xticks(range(1, n_units+1))
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Add text with overall statistics
        overall_stats = f"Overall Mean Rate: {np.mean(mean_rates):.2f} Hz\nStd: {np.std(mean_rates):.2f} Hz"
        axes[0].text(0.02, 0.95, overall_stats, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 2. Plot coefficient of variation for all motor units
        axes[1].bar(range(1, n_units+1), cvs, alpha=0.7, color='orange')
        axes[1].set_xlabel('Motor Unit')
        axes[1].set_ylabel('Coefficient of Variation')
        axes[1].set_title('Discharge Variability by Motor Unit')
        axes[1].set_xticks(range(1, n_units+1))
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Add text with overall statistics
        overall_cv_stats = f"Overall Mean CV: {np.mean(cvs):.3f}\nStd: {np.std(cvs):.3f}"
        axes[1].text(0.02, 0.95, overall_cv_stats, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 3. Plot reference signal if it exists
        if has_ref_signal:
            ref_signal = data['ref_signal'].flatten()

            # Create time vector for reference signal
            ref_time = np.arange(len(ref_signal)) / FSAMP  # Convert to seconds

            # Plot reference signal
            axes[2].plot(ref_time, ref_signal, 'k-', linewidth=2)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Force (N)')
            axes[2].set_title('Reference Signal (Force)')
            axes[2].grid(True, linestyle='--', alpha=0.7)

            # Calculate and display statistics for reference signal
            mean_force = np.mean(ref_signal)
            std_force = np.std(ref_signal)
            cv_force = std_force / mean_force if mean_force != 0 else 0

            # Add text with statistics
            force_stats = f"Mean Force: {mean_force:.2f} N\nStd: {std_force:.2f} N\nCV: {cv_force:.3f}"
            axes[2].text(0.02, 0.95, force_stats, transform=axes[2].transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Set title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(file_path.name, fontsize=16)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def plot_combined_analysis(file_path, title=None):
    """
    Create a combined plot showing ISI, CV, and reference signal for all motor units

    Parameters:
    -----------
    file_path : Path
        Path to the .mat file
    title : str, optional
        Title for the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    try:
        # Load the .mat file
        data = sio.loadmat(file_path)

        # Check if MUPulses exists in the data
        if 'MUPulses' not in data:
            print(f"MUPulses not found in {file_path}")
            return None

        # Get the MUPulses data
        mu_pulses = data['MUPulses']
        n_units = mu_pulses.shape[1]

        # Check if reference signal exists
        has_ref_signal = 'ref_signal' in data

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))

        # Define grid layout
        if has_ref_signal:
            # With reference signal: 3 rows, 2 columns
            gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

            # ISI histograms for all units (top left)
            ax_isi = fig.add_subplot(gs[0, 0])

            # CV plot (top right)
            ax_cv = fig.add_subplot(gs[0, 1])

            # Firing rates over time (middle row, spans both columns)
            ax_rates = fig.add_subplot(gs[1, :])

            # Reference signal (bottom row, spans both columns)
            ax_ref = fig.add_subplot(gs[2, :])
        else:
            # Without reference signal: 2 rows, 2 columns
            gs = plt.GridSpec(2, 2, figure=fig)

            # ISI histograms for all units (top left)
            ax_isi = fig.add_subplot(gs[0, 0])

            # CV plot (top right)
            ax_cv = fig.add_subplot(gs[0, 1])

            # Firing rates over time (bottom row, spans both columns)
            ax_rates = fig.add_subplot(gs[1, :])

        # Process each motor unit
        mean_rates = []
        cvs = []
        all_differences = []
        colors = plt.cm.tab10.colors  # Color cycle

        for i in range(n_units):
            # Get pulses for this motor unit
            pulses = mu_pulses[0, i]

            # Calculate differences
            differences = calculate_discharge_differences(pulses)
            all_differences.append(differences)

            # Calculate mean rate and CV
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            cv = std_diff / mean_diff

            # Convert to Hz
            mean_rate = 1000 / mean_diff

            mean_rates.append(mean_rate)
            cvs.append(cv)

            # Calculate and plot instantaneous firing rate
            times, rates = calculate_instantaneous_rate(pulses, window_size=1000)
            if len(times) > 0:
                color = colors[i % len(colors)]
                ax_rates.plot(times/1000, rates, color=color, label=f"MU {i+1}")

        # 1. Plot combined ISI histogram
        bins = np.linspace(0, 200, 50)  # 0-200 ms range with 50 bins
        for i, differences in enumerate(all_differences):
            color = colors[i % len(colors)]
            ax_isi.hist(differences, bins=bins, alpha=0.3, color=color, label=f"MU {i+1}")

        ax_isi.set_xlabel('Inter-spike Interval (ms)')
        ax_isi.set_ylabel('Count')
        ax_isi.set_title('ISI Distributions')
        ax_isi.legend(loc='upper right')
        ax_isi.grid(True, linestyle='--', alpha=0.7)

        # 2. Plot CV for each motor unit
        ax_cv.bar(range(1, n_units+1), cvs, alpha=0.7, color=colors[:n_units])
        ax_cv.set_xlabel('Motor Unit')
        ax_cv.set_ylabel('Coefficient of Variation')
        ax_cv.set_title('Discharge Variability')
        ax_cv.set_xticks(range(1, n_units+1))
        ax_cv.grid(True, linestyle='--', alpha=0.7)

        # Add text with overall statistics
        overall_cv_stats = f"Mean CV: {np.mean(cvs):.3f}\nStd: {np.std(cvs):.3f}"
        ax_cv.text(0.02, 0.95, overall_cv_stats, transform=ax_cv.transAxes,
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 3. Firing rates plot (already populated in the loop)
        ax_rates.set_xlabel('Time (s)')
        ax_rates.set_ylabel('Firing Rate (Hz)')
        ax_rates.set_title('Instantaneous Firing Rates')
        ax_rates.grid(True, linestyle='--', alpha=0.7)
        ax_rates.legend(loc='upper right')

        # 4. Reference signal if it exists
        if has_ref_signal:
            ref_signal = data['ref_signal'].flatten()

            # Create time vector for reference signal
            ref_time = np.arange(len(ref_signal)) / FSAMP  # Convert to seconds

            # Plot reference signal
            ax_ref.plot(ref_time, ref_signal, 'k-', linewidth=2)
            ax_ref.set_xlabel('Time (s)')
            ax_ref.set_ylabel('Force (N)')
            ax_ref.set_title('Reference Signal (Force)')
            ax_ref.grid(True, linestyle='--', alpha=0.7)

            # Calculate and display statistics for reference signal
            mean_force = np.mean(ref_signal)
            std_force = np.std(ref_signal)
            cv_force = std_force / mean_force if mean_force != 0 else 0

            # Add text with statistics
            force_stats = f"Mean Force: {mean_force:.2f} N\nStd: {std_force:.2f} N\nCV: {cv_force:.3f}"
            ax_ref.text(0.02, 0.95, force_stats, transform=ax_ref.transAxes,
                      verticalalignment='top', horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Set title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(file_path.name, fontsize=16)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def generate_fdi_plateau_spikes_csv(mat_file_path, output_file_path=None, plateau_info=None):
    """
    Generate a CSV file with spike times from plateau phases for FDI data

    Parameters:
    -----------
    mat_file_path : Path or str
        Path to the .mat file containing FDI spike data
    output_file_path : Path or str, optional
        Path where to save the CSV file. If None, a default name will be used
    plateau_info : dict, optional
        Dictionary with plateau information. If None, will try to extract from fdi_plateaus.csv

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

        # Set muscle to FDI
        muscle = 'FDI'

        # Extract force level from filename or data
        force_level = None

        # Try to extract from filename
        file_name = mat_file_path.name
        rel_path = str(mat_file_path.relative_to(DATA_PATH)) if mat_file_path.is_relative_to(DATA_PATH) else mat_file_path.name

        # Try to extract force level from filename
        force_match = re.search(r'_(\d+)_', file_name)
        if force_match:
            force_level = int(force_match.group(1))
        elif re.search(r'(\d+)MVC', file_name, re.IGNORECASE):
            # Try to match patterns like "10MVC" or "10mvc"
            mvc_match = re.search(r'(\d+)MVC', file_name, re.IGNORECASE)
            force_level = int(mvc_match.group(1))
        elif 'F1' in file_name:
            # For files with F1 pattern, assume 10% MVC
            force_level = 10

        # If not found in filename, try to extract from data
        if force_level is None and 'force_level' in data:
            force_level = int(data['force_level'][0])

        # Default value if still not found
        if force_level is None:
            force_level = 0

        # Determine plateau information
        plateaus = {}

        # If plateau_info is provided, use it
        if plateau_info is not None:
            plateaus = plateau_info
        else:
            # Try to load from the fdi_plateaus.csv file
            plateau_csv = Path("Notebooks/marimo_notebooks/fdi_plateaus.csv")
            if not plateau_csv.exists():
                print(f"fdi_plateaus.csv not found at {plateau_csv}")
                return None

            plateau_df = pd.read_csv(plateau_csv)

            # Find the row for this file
            # First try with the full relative path
            file_row = plateau_df[plateau_df['filename'] == rel_path]

            # If not found, try with just the filename
            if file_row.empty:
                file_row = plateau_df[plateau_df['filename'].str.endswith(file_name)]

            # If still not found, try with the parent folder + filename
            if file_row.empty and mat_file_path.parent.name:
                parent_file = f"{mat_file_path.parent.name}/{file_name}"
                file_row = plateau_df[plateau_df['filename'] == parent_file]

            if file_row.empty:
                print(f"No plateau information found for {rel_path} in fdi_plateaus.csv")
                return None

            # Extract plateau information
            for i in range(1, 5):  # Assuming up to 4 plateaus
                if f'plateau{i}_start' in file_row.columns and f'plateau{i}_end' in file_row.columns:
                    start = file_row[f'plateau{i}_start'].values[0]
                    end = file_row[f'plateau{i}_end'].values[0]
                    if not pd.isna(start) and not pd.isna(end):
                        # Convert from seconds to sample points
                        plateaus[i-1] = (start * FSAMP, end * FSAMP)

        # If no plateaus found, return None
        if not plateaus:
            print(f"No plateau information available for {mat_file_path}")
            return None

        # Create results directory outside the data folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Create subdirectory for spike CSV files
        spike_csv_dir = results_dir / "spike_csv"
        spike_csv_dir.mkdir(exist_ok=True)

        # Determine output file path
        if output_file_path is None:
            # Create a default name
            output_file_path = spike_csv_dir / f"{mat_file_path.stem}_all_plateaus_spikes.csv"

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
            # Write exactly 6 header comment lines
            csvfile.write(f"# File: {mat_file_path.name}\n")
            csvfile.write(f"# Muscle: {muscle}\n")
            csvfile.write(f"# Force Level: {force_level}% MVC\n")

            # Write plateau information
            plateau_info_str = "# Plateaus: "
            for plateau_idx, (start_time, end_time) in plateaus.items():
                start_sec = start_time / FSAMP
                end_sec = end_time / FSAMP
                plateau_info_str += f"{plateau_idx+1}: {start_sec:.2f}s-{end_sec:.2f}s, "

            # Remove trailing comma and space
            plateau_info_str = plateau_info_str.rstrip(", ")
            csvfile.write(f"{plateau_info_str}\n")

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

def process_fdi_files_to_csv():
    """
    Process FDI .mat files that have plateau information in fdi_plateaus.csv
    and generate CSV files with spike times from plateau phases

    Returns:
    --------
    generated_files : list
        List of generated CSV files
    """
    # Load the plateau information from fdi_plateaus.csv
    plateau_csv = Path("Notebooks/marimo_notebooks/fdi_plateaus.csv")
    if not plateau_csv.exists():
        print(f"fdi_plateaus.csv not found at {plateau_csv}")
        return []

    plateau_df = pd.read_csv(plateau_csv)

    # Path to FDI folder
    fdi_path = DATA_PATH / "FDI"

    if not fdi_path.exists():
        print("FDI folder not found")
        return []

    # Get all subfolders in FDI
    subfolders = list_subfolders(fdi_path)

    if not subfolders:
        print("No subfolders found in FDI folder")
        return []

    # Process only files listed in fdi_plateaus.csv
    generated_files = []

    # Process each file in the plateau CSV
    for _, row in plateau_df.iterrows():
        filename = row['filename']

        # Extract folder and file parts
        if '/' in filename:
            folder_name, file_name = filename.split('/', 1)
        else:
            folder_name = None
            file_name = filename

        # Find the file in the FDI folder
        found = False

        # If folder is specified, look in that specific subfolder
        if folder_name:
            for subfolder in subfolders:
                if subfolder.name == folder_name:
                    mat_file = subfolder / file_name
                    if mat_file.exists():
                        found = True
                        print(f"Processing {mat_file}...")
                        output_file = generate_fdi_plateau_spikes_csv(mat_file)
                        if output_file:
                            generated_files.append(output_file)
                    break

        # If not found or folder not specified, search all subfolders
        if not found:
            for subfolder in subfolders:
                for mat_file in subfolder.glob("*.mat"):
                    if mat_file.name == file_name:
                        found = True
                        print(f"Processing {mat_file}...")
                        output_file = generate_fdi_plateau_spikes_csv(mat_file)
                        if output_file:
                            generated_files.append(output_file)
                        break
                if found:
                    break

        if not found:
            print(f"Could not find file {filename} in FDI folder")

    # Return results
    if generated_files:
        print(f"\nSuccessfully generated {len(generated_files)} CSV files:")
        for f in generated_files:
            print(f"- {f}")
    else:
        print("No CSV files were generated. Check the console for errors.")

    return generated_files

def plot_all_fdi_discharge_differences():
    """
    Plot discharge differences for all FDI files
    """
    # Path to FDI folder
    fdi_path = DATA_PATH / "FDI"

    if not fdi_path.exists():
        print("FDI folder not found")
        return

    # Get all subfolders in FDI
    subfolders = list_subfolders(fdi_path)

    # Create output directories for plots
    detail_dir = Path("FDI_discharge_plots")
    detail_dir.mkdir(exist_ok=True)

    summary_dir = Path("FDI_summary_plots")
    summary_dir.mkdir(exist_ok=True)

    combined_dir = Path("FDI_combined_plots")
    combined_dir.mkdir(exist_ok=True)

    # Process each subfolder
    for subfolder in subfolders:
        print(f"Processing {subfolder.name}...")

        # Get all .mat files in this subfolder
        mat_files = list_mat_files(subfolder)

        # Process each .mat file
        for mat_file in mat_files:
            print(f"  Processing {mat_file.name}...")

            # 1. Plot detailed discharge differences
            fig_detail = plot_discharge_differences(mat_file, title=f"{subfolder.name} - {mat_file.name}")

            if fig_detail:
                # Save the figure
                output_file = detail_dir / f"{subfolder.name}_{mat_file.stem}_detail.png"
                fig_detail.savefig(output_file, dpi=300)
                plt.close(fig_detail)
                print(f"  Saved detailed plot to {output_file}")

            # 2. Plot summary comparison of all motor units
            fig_summary = plot_mu_summary(mat_file, title=f"{subfolder.name} - {mat_file.name} - Summary")

            if fig_summary:
                # Save the figure
                output_file = summary_dir / f"{subfolder.name}_{mat_file.stem}_summary.png"
                fig_summary.savefig(output_file, dpi=300)
                plt.close(fig_summary)
                print(f"  Saved summary plot to {output_file}")

            # 3. Plot combined analysis
            fig_combined = plot_combined_analysis(mat_file, title=f"{subfolder.name} - {mat_file.name} - Combined Analysis")

            if fig_combined:
                # Save the figure
                output_file = combined_dir / f"{subfolder.name}_{mat_file.stem}_combined.png"
                fig_combined.savefig(output_file, dpi=300)
                plt.close(fig_combined)
                print(f"  Saved combined plot to {output_file}")

if __name__ == "__main__":
    print("Motoneuron Data Explorer")
    print("=======================")
    print("\nOptions:")
    print("1. Explore data interactively")
    print("2. Plot discharge differences for all FDI files")
    print("3. Generate CSV files with FDI spike times from plateau phases")

    option = input("\nSelect an option (1-3): ")

    if option == '1':
        explore_data_interactive()
    elif option == '2':
        print("\nFDI Discharge Differences Analyzer")
        print("==================================")
        print("\nThis program automatically plots the differences between consecutive discharges")
        print("for all motor units in all FDI files.")

        # Plot discharge differences for all FDI files
        plot_all_fdi_discharge_differences()
    elif option == '3':
        print("\nFDI Plateau Spikes CSV Generator")
        print("===============================")
        print("\nThis program generates CSV files with spike times from plateau phases for FDI data.")
        print("It will process all .mat files in the FDI folder and its subfolders.")

        # Process FDI files
        process_fdi_files_to_csv()
    else:
        print("Invalid option")
