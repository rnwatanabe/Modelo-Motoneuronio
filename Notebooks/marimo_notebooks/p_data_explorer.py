import sys

sys.path.append("./../")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from pathlib import Path
import os
import csv

# set backend to QtAgg (interactive GUI backend)
plt.switch_backend("Agg")

# Constants from README
FSAMP = 10240  # Sampling frequency for all datasets

# Data folder path
DATA_PATH = Path("data")

def list_mat_files(folder_path, pattern=None):
    """
    List all .mat files in a given folder, optionally filtering by pattern

    Parameters:
    -----------
    folder_path : Path
        Path to the folder
    pattern : str, optional
        Pattern to filter files by

    Returns:
    --------
    mat_files : list
        List of .mat files
    """
    mat_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.mat']

    if pattern:
        mat_files = [f for f in mat_files if pattern in f.name]

    return mat_files

def load_ramp_definitions(folder_path):
    """
    Load ramp definitions from RampDefinition_xx.mat files

    Parameters:
    -----------
    folder_path : Path
        Path to the folder containing RampDefinition_xx.mat files

    Returns:
    --------
    ramp_defs : dict
        Dictionary with force levels as keys and another dictionary as values.
        The inner dictionary has plateau indices (0, 1, 2, 3) as keys and (start, stop) tuples as values.
    """
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
                    print(f"Loaded ramp definition for {force_level}% MVC, plateau {i+1}: {start_idx/FSAMP:.2f}s to {stop_idx/FSAMP:.2f}s")
            else:
                print(f"Warning: startMax or stopMax not found in {file_path}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return ramp_defs

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

def calculate_force_cv(data, plateau_start, plateau_stop):
    """
    Calculate the coefficient of variation (CV) of force during a plateau phase

    Parameters:
    -----------
    data : dict
        Dictionary containing the data loaded from a .mat file
    plateau_start : int
        Start index of the plateau phase
    plateau_stop : int
        Stop index of the plateau phase

    Returns:
    --------
    cv : float
        Coefficient of variation of force during the plateau phase
    mean_force : float
        Mean force during the plateau phase
    std_force : float
        Standard deviation of force during the plateau phase
    """
    # Check if reference signal (force) exists
    if 'ref_signal' not in data:
        return None, None, None

    # Get the reference signal (force)
    ref_signal = data['ref_signal'].flatten()

    # Extract the plateau portion
    plateau_force = ref_signal[plateau_start:plateau_stop]

    # Calculate statistics
    mean_force = np.mean(plateau_force)
    std_force = np.std(plateau_force)
    cv = std_force / mean_force if mean_force > 0 else 0

    return cv, mean_force, std_force

def plot_cv_vs_rate_plateau(folder_path, muscle_type, ramp_defs):
    """
    Create a plot with CV on x-axis and firing rate on y-axis, with quadratic fit
    Using data from the plateau phase defined in ramp_defs

    Parameters:
    -----------
    folder_path : Path
        Path to the folder containing data files
    muscle_type : str
        Muscle type (VL or VM)
    ramp_defs : dict
        Dictionary with force levels as keys and (start, stop) tuples as values

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create output directory for plots
    cv_rate_dir = Path(f"{folder_path.name}_{muscle_type}_CV_Rate_Analysis")
    cv_rate_dir.mkdir(exist_ok=True)

    # Get all files for the specified muscle type
    muscle_files = list_mat_files(folder_path, f"{muscle_type}_")

    # Lists to store data from all motor units
    all_cvs = []
    all_rates = []
    all_force_levels = []  # To color-code by force level
    all_force_cvs = []     # To store force CV for each plateau

    # Process all files
    for file_path in muscle_files:
        try:
            # Extract force level from filename (e.g., VL_05.mat -> 5)
            force_level = int(file_path.stem.split('_')[1])

            # Check if we have ramp definition for this force level
            if force_level not in ramp_defs:
                print(f"Warning: No ramp definition for {force_level}% MVC, skipping {file_path}")
                continue

            # Get plateau start and stop indices
            plateau_start, plateau_stop = ramp_defs[force_level]

            # Load the data
            data = sio.loadmat(file_path)
            if 'MUPulses' not in data:
                print(f"Warning: MUPulses not found in {file_path}")
                continue

            # Calculate force CV for this plateau
            force_cv, mean_force, std_force = calculate_force_cv(data, plateau_start, plateau_stop)

            # If force data is available, print information
            if force_cv is not None:
                print(f"  Force CV for {file_path.name} (plateau): CV={force_cv:.3f}, Mean={mean_force:.2f}N, SD={std_force:.2f}N")

            # Get the MUPulses data
            mu_pulses = data['MUPulses']

            # Process each motor unit
            for i in range(mu_pulses.shape[1]):
                pulses = mu_pulses[0, i]

                # Filter pulses to only include those in the plateau phase
                plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                # Skip if not enough pulses in plateau
                if len(plateau_pulses) < 5:
                    print(f"  Skipping MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                    continue

                # Calculate interspike intervals for plateau pulses
                differences = calculate_discharge_differences(plateau_pulses)

                # Calculate statistics
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                cv = std_diff / mean_diff

                # Convert to Hz
                mean_rate = 1000 / mean_diff

                # Store the data
                all_cvs.append(cv)
                all_rates.append(mean_rate)
                all_force_levels.append(force_level)
                all_force_cvs.append(force_cv if force_cv is not None else 0)

                print(f"  Added MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to numpy arrays for easier manipulation
    all_cvs = np.array(all_cvs)
    all_rates = np.array(all_rates)
    all_force_levels = np.array(all_force_levels)
    all_force_cvs = np.array(all_force_cvs)

    # Create the CV vs Rate plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with color coding by force level
    scatter = ax.scatter(all_cvs, all_rates, c=all_force_levels,
                         cmap='viridis', alpha=0.7, s=80, edgecolors='k')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Force Level (% MVC)')

    # Separate data by force level
    force_levels = np.unique(all_force_levels)

    # Colors for different force levels
    colors = ['g', 'b', 'r']

    # Fit quadratic curves for each force level
    for i, level in enumerate(force_levels):
        level_indices = all_force_levels == level
        level_cvs = all_cvs[level_indices]
        level_rates = all_rates[level_indices]
        level_force_cvs = all_force_cvs[level_indices]

        if len(level_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(level_cvs, level_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(level_cvs), max(level_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{level}% MVC Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(level_rates)
            ss_tot = np.sum((level_rates - y_mean) ** 2)
            ss_res = np.sum((level_rates - poly(level_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{level}% MVC R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title(f'Relationship Between Firing Rate and Discharge Variability in {muscle_type} Motor Units\n'
                f'Plateau Phase - Separate Quadratic Fits for Each Force Level')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add text with statistics for each force level
    stats_text = ""
    for level in force_levels:
        level_indices = all_force_levels == level
        level_cvs = all_cvs[level_indices]
        level_rates = all_rates[level_indices]
        level_force_cvs = all_force_cvs[level_indices]

        # Calculate mean force CV if available
        mean_force_cv = np.mean(level_force_cvs[level_force_cvs > 0]) if np.any(level_force_cvs > 0) else 0

        stats_text += (f"{level}% MVC (n={len(level_cvs)}):\n"
                      f"Mean CV: {np.mean(level_cvs):.3f} ± {np.std(level_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(level_rates):.2f} ± {np.std(level_rates):.2f} Hz\n"
                      f"Force CV: {mean_force_cv:.3f}\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = cv_rate_dir / f"{folder_path.name}_{muscle_type}_CV_vs_Rate_plateau_quadratic_fits.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved CV vs Rate plot (plateau) to {output_file}")

    return fig

def plot_cv_vs_rate_whole_data(folder_path, muscle_type):
    """
    Create a plot with CV on x-axis and firing rate on y-axis, with quadratic fit
    Using the whole data (not just plateau phases)

    Parameters:
    -----------
    folder_path : Path
        Path to the folder containing data files
    muscle_type : str
        Muscle type (VL or VM)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create output directory for plots
    cv_rate_dir = Path(f"{folder_path.name}_{muscle_type}_CV_Rate_Analysis")
    cv_rate_dir.mkdir(exist_ok=True)

    # Get all files for the specified muscle type
    muscle_files = list_mat_files(folder_path, f"{muscle_type}_")

    # Lists to store data from all motor units
    all_cvs = []
    all_rates = []
    all_force_levels = []  # To color-code by force level

    # Process all files
    for file_path in muscle_files:
        try:
            # Extract force level from filename (e.g., VL_05.mat -> 5)
            force_level = int(file_path.stem.split('_')[1])

            # Load the data
            data = sio.loadmat(file_path)
            if 'MUPulses' not in data:
                print(f"Warning: MUPulses not found in {file_path}")
                continue

            # Get the MUPulses data
            mu_pulses = data['MUPulses']

            # Process each motor unit
            for i in range(mu_pulses.shape[1]):
                pulses = mu_pulses[0, i]

                # Calculate interspike intervals for all pulses
                differences = calculate_discharge_differences(pulses)

                # Calculate statistics
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                cv = std_diff / mean_diff

                # Convert to Hz
                mean_rate = 1000 / mean_diff

                # Store the data
                all_cvs.append(cv)
                all_rates.append(mean_rate)
                all_force_levels.append(force_level)

                print(f"  Added MU {i+1} from {file_path.name} (whole data): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to numpy arrays for easier manipulation
    all_cvs = np.array(all_cvs)
    all_rates = np.array(all_rates)
    all_force_levels = np.array(all_force_levels)

    # Create the CV vs Rate plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot with color coding by force level
    scatter = ax.scatter(all_cvs, all_rates, c=all_force_levels,
                         cmap='viridis', alpha=0.7, s=80, edgecolors='k')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Force Level (% MVC)')

    # Separate data by force level
    force_levels = np.unique(all_force_levels)

    # Colors for different force levels
    colors = ['g', 'b', 'r']

    # Fit quadratic curves for each force level
    for i, level in enumerate(force_levels):
        level_indices = all_force_levels == level
        level_cvs = all_cvs[level_indices]
        level_rates = all_rates[level_indices]

        if len(level_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(level_cvs, level_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(level_cvs), max(level_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{level}% MVC Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(level_rates)
            ss_tot = np.sum((level_rates - y_mean) ** 2)
            ss_res = np.sum((level_rates - poly(level_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{level}% MVC R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title(f'Relationship Between Firing Rate and Discharge Variability in {muscle_type} Motor Units\n'
                f'Whole Data - Separate Quadratic Fits for Each Force Level')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add text with statistics for each force level
    stats_text = ""
    for level in force_levels:
        level_indices = all_force_levels == level
        level_cvs = all_cvs[level_indices]
        level_rates = all_rates[level_indices]

        stats_text += (f"{level}% MVC (n={len(level_cvs)}):\n"
                      f"Mean CV: {np.mean(level_cvs):.3f} ± {np.std(level_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(level_rates):.2f} ± {np.std(level_rates):.2f} Hz\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = cv_rate_dir / f"{folder_path.name}_{muscle_type}_CV_vs_Rate_whole_data_quadratic_fits.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved CV vs Rate plot (whole data) to {output_file}")

    return fig

def plot_force_with_plateaus(data, ramp_defs, force_level, file_path, muscle_type):
    """
    Plot force signal with highlighted plateaus and CV information

    Parameters:
    -----------
    data : dict
        Dictionary containing the data loaded from a .mat file
    ramp_defs : dict
        Dictionary with force levels as keys and (start, stop) tuples as values
    force_level : int
        Force level (% MVC)
    file_path : Path
        Path to the .mat file
    muscle_type : str
        Muscle type (VL or VM)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Check if reference signal (force) exists
    if 'ref_signal' not in data:
        return None

    # Get the reference signal (force)
    ref_signal = data['ref_signal'].flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot force signal
    time = np.arange(len(ref_signal)) / FSAMP
    ax.plot(time, ref_signal, 'k-', linewidth=1.5, label='Force')

    # Process each plateau
    plateau_colors = ['g', 'b', 'r', 'c', 'm', 'y']

    # Check if ramp_defs is a dictionary of dictionaries (multiple plateaus)
    if isinstance(next(iter(ramp_defs.values())), dict):
        # Multiple plateaus case
        for plateau_idx, (start_idx, stop_idx) in ramp_defs[force_level].items():
            color = plateau_colors[plateau_idx % len(plateau_colors)]
            start_time = start_idx / FSAMP
            stop_time = stop_idx / FSAMP

            # Highlight plateau region
            ax.axvspan(start_time, stop_time, alpha=0.2, color=color,
                      label=f'Plateau {plateau_idx+1}' if plateau_idx == 0 else "")

            # Calculate force CV for this plateau
            plateau_signal = ref_signal[start_idx:stop_idx+1]
            if len(plateau_signal) > 1:
                mean = np.mean(plateau_signal)
                std = np.std(plateau_signal)
                cv = std / mean if mean != 0 else np.nan

                # Add CV text directly on the plot
                ax.text((start_time+stop_time)/2, np.max(ref_signal)*0.95,
                        f'CV: {cv:.3f}', color=color, ha='center', va='top',
                        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Add vertical lines at plateau boundaries
            ax.axvline(x=start_time, color=color, linestyle='--', alpha=0.7)
            ax.axvline(x=stop_time, color=color, linestyle=':', alpha=0.7)
    else:
        # Single plateau case (old format)
        plateau_start, plateau_stop = ramp_defs[force_level]
        start_time = plateau_start / FSAMP
        stop_time = plateau_stop / FSAMP

        # Highlight plateau region
        ax.axvspan(start_time, stop_time, alpha=0.2, color='green', label='Plateau')

        # Calculate force CV for the plateau
        force_cv, mean_force, std_force = calculate_force_cv(data, plateau_start, plateau_stop)

        # Add CV information to the plot
        if force_cv is not None:
            # Add text with force CV information
            cv_text = f"Plateau Force CV: {force_cv:.3f}\nMean: {mean_force:.2f}N\nSD: {std_force:.2f}N"
            ax.text(0.02, 0.95, cv_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Also add CV directly on the plateau for consistency
            ax.text((start_time+stop_time)/2, np.max(ref_signal)*0.95,
                    f'CV: {force_cv:.3f}', color='green', ha='center', va='top',
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title(f'{file_path.stem} - {muscle_type} - {force_level}% MVC')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()

    return fig

def plot_combined_analysis():
    """
    Create a combined plot with data from all P folders, differentiating by muscle type
    Only using data from the plateau phases
    """
    print("\nCreating combined analysis plot for all P folders (plateau phases only)...")

    # Create output directory for plots
    combined_dir = Path("P_Combined_Analysis")
    combined_dir.mkdir(exist_ok=True)

    # Create directory for force plots
    force_plots_dir = combined_dir / "Force_Plots"
    force_plots_dir.mkdir(exist_ok=True)

    # Lists to store data from all motor units
    all_cvs = []
    all_rates = []
    all_muscle_types = []  # To differentiate by muscle type
    all_force_levels = []  # To color-code by force level
    all_subjects = []      # To track which subject (P01 or P02)
    all_force_cvs = []     # To store force CV for each plateau

    # Process P01_iEMG_Ramps
    p01_path = DATA_PATH / "P01_iEMG_Ramps"
    if p01_path.exists():
        print(f"  Processing {p01_path.name}...")

        # Load ramp definitions (plateau phases)
        p01_ramp_defs = load_ramp_definitions(p01_path)
        if not p01_ramp_defs:
            print(f"  No ramp definitions found in {p01_path}, skipping...")
            return

        # Process VL data
        vl_files = list_mat_files(p01_path, "VL_")
        for file_path in vl_files:
            try:
                # Extract force level from filename (e.g., VL_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])

                # Check if we have ramp definition for this force level
                if force_level not in p01_ramp_defs:
                    print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                    continue

                # Get plateau start and stop indices
                plateau_start, plateau_stop = p01_ramp_defs[force_level]

                # Load the data
                data = sio.loadmat(file_path)
                if 'MUPulses' not in data:
                    continue

                # Calculate force CV for this plateau
                force_cv, mean_force, std_force = calculate_force_cv(data, plateau_start, plateau_stop)

                # If force data is available, print information and create force plot
                if force_cv is not None:
                    print(f"  Force CV for {file_path.name} (plateau): CV={force_cv:.3f}, Mean={mean_force:.2f}N, SD={std_force:.2f}N")

                    # Create and save force plot
                    force_fig = plot_force_with_plateaus(data, p01_ramp_defs, force_level, file_path, "VL")
                    if force_fig:
                        force_output_file = force_plots_dir / f"P01_VL_{force_level}_force_plateau.png"
                        force_fig.savefig(force_output_file, dpi=300)
                        plt.close(force_fig)
                        print(f"  Saved force plot to {force_output_file}")

                # Get the MUPulses data
                mu_pulses = data['MUPulses']

                # Process each motor unit
                for i in range(mu_pulses.shape[1]):
                    pulses = mu_pulses[0, i]

                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                    # Skip if not enough pulses in plateau
                    if len(plateau_pulses) < 5:
                        print(f"    Skipping P01 VL MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                        continue

                    # Calculate interspike intervals for plateau pulses
                    differences = calculate_discharge_differences(plateau_pulses)

                    # Calculate statistics
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences)
                    cv = std_diff / mean_diff

                    # Convert to Hz
                    mean_rate = 1000 / mean_diff

                    # Store the data
                    all_cvs.append(cv)
                    all_rates.append(mean_rate)
                    all_muscle_types.append("VL")
                    all_force_levels.append(force_level)
                    all_subjects.append("P01")
                    all_force_cvs.append(force_cv if force_cv is not None else 0)

                    print(f"    Added P01 VL MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

        # Process VM data
        vm_files = list_mat_files(p01_path, "VM_")
        for file_path in vm_files:
            try:
                # Extract force level from filename (e.g., VM_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])

                # Check if we have ramp definition for this force level
                if force_level not in p01_ramp_defs:
                    print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                    continue

                # Get plateau start and stop indices
                plateau_start, plateau_stop = p01_ramp_defs[force_level]

                # Load the data
                data = sio.loadmat(file_path)
                if 'MUPulses' not in data:
                    continue

                # Calculate force CV for this plateau
                force_cv, mean_force, std_force = calculate_force_cv(data, plateau_start, plateau_stop)

                # If force data is available, print information and create force plot
                if force_cv is not None:
                    print(f"  Force CV for {file_path.name} (plateau): CV={force_cv:.3f}, Mean={mean_force:.2f}N, SD={std_force:.2f}N")

                    # Create and save force plot
                    force_fig = plot_force_with_plateaus(data, p01_ramp_defs, force_level, file_path, "VM")
                    if force_fig:
                        force_output_file = force_plots_dir / f"P01_VM_{force_level}_force_plateau.png"
                        force_fig.savefig(force_output_file, dpi=300)
                        plt.close(force_fig)
                        print(f"  Saved force plot to {force_output_file}")

                # Get the MUPulses data
                mu_pulses = data['MUPulses']

                # Process each motor unit
                for i in range(mu_pulses.shape[1]):
                    pulses = mu_pulses[0, i]

                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                    # Skip if not enough pulses in plateau
                    if len(plateau_pulses) < 5:
                        print(f"    Skipping P01 VM MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                        continue

                    # Calculate interspike intervals for plateau pulses
                    differences = calculate_discharge_differences(plateau_pulses)

                    # Calculate statistics
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences)
                    cv = std_diff / mean_diff

                    # Convert to Hz
                    mean_rate = 1000 / mean_diff

                    # Store the data
                    all_cvs.append(cv)
                    all_rates.append(mean_rate)
                    all_muscle_types.append("VM")
                    all_force_levels.append(force_level)
                    all_subjects.append("P01")
                    all_force_cvs.append(force_cv if force_cv is not None else 0)

                    print(f"    Added P01 VM MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

    # Process P02_iEMG_Ramps
    p02_path = DATA_PATH / "P02_iEMG_Ramps"
    if p02_path.exists():
        print(f"  Processing {p02_path.name}...")

        # Load ramp definitions (plateau phases)
        p02_ramp_defs = load_ramp_definitions(p02_path)
        if not p02_ramp_defs:
            print(f"  No ramp definitions found in {p02_path}, skipping...")
            return

        # Process VL data
        vl_files = list_mat_files(p02_path, "VL_")
        for file_path in vl_files:
            try:
                # Extract force level from filename (e.g., VL_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])

                # Check if we have ramp definition for this force level
                if force_level not in p02_ramp_defs:
                    print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                    continue

                # Get plateau start and stop indices
                plateau_start, plateau_stop = p02_ramp_defs[force_level]

                # Load the data
                data = sio.loadmat(file_path)
                if 'MUPulses' not in data:
                    continue

                # Get the MUPulses data
                mu_pulses = data['MUPulses']

                # Process each motor unit
                for i in range(mu_pulses.shape[1]):
                    pulses = mu_pulses[0, i]

                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                    # Skip if not enough pulses in plateau
                    if len(plateau_pulses) < 5:
                        print(f"    Skipping P02 VL MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                        continue

                    # Calculate interspike intervals for plateau pulses
                    differences = calculate_discharge_differences(plateau_pulses)

                    # Calculate statistics
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences)
                    cv = std_diff / mean_diff

                    # Convert to Hz
                    mean_rate = 1000 / mean_diff

                    # Store the data
                    all_cvs.append(cv)
                    all_rates.append(mean_rate)
                    all_muscle_types.append("VL")
                    all_force_levels.append(force_level)
                    all_subjects.append("P02")

                    print(f"    Added P02 VL MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

        # Process VM data
        vm_files = list_mat_files(p02_path, "VM_")
        for file_path in vm_files:
            try:
                # Extract force level from filename (e.g., VM_05.mat -> 5)
                force_level = int(file_path.stem.split('_')[1])

                # Check if we have ramp definition for this force level
                if force_level not in p02_ramp_defs:
                    print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                    continue

                # Get plateau start and stop indices
                plateau_start, plateau_stop = p02_ramp_defs[force_level]

                # Load the data
                data = sio.loadmat(file_path)
                if 'MUPulses' not in data:
                    continue

                # Get the MUPulses data
                mu_pulses = data['MUPulses']

                # Process each motor unit
                for i in range(mu_pulses.shape[1]):
                    pulses = mu_pulses[0, i]

                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                    # Skip if not enough pulses in plateau
                    if len(plateau_pulses) < 5:
                        print(f"    Skipping P02 VM MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                        continue

                    # Calculate interspike intervals for plateau pulses
                    differences = calculate_discharge_differences(plateau_pulses)

                    # Calculate statistics
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences)
                    cv = std_diff / mean_diff

                    # Convert to Hz
                    mean_rate = 1000 / mean_diff

                    # Store the data
                    all_cvs.append(cv)
                    all_rates.append(mean_rate)
                    all_muscle_types.append("VM")
                    all_force_levels.append(force_level)
                    all_subjects.append("P02")

                    print(f"    Added P02 VM MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

    # Convert to numpy arrays for easier manipulation
    all_cvs = np.array(all_cvs)
    all_rates = np.array(all_rates)
    all_muscle_types = np.array(all_muscle_types)
    all_force_levels = np.array(all_force_levels)
    all_subjects = np.array(all_subjects)

    # Create the combined CV vs Rate plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with different markers for different muscle types
    # and color coding by force level
    for muscle_type in np.unique(all_muscle_types):
        mask = all_muscle_types == muscle_type
        scatter = ax.scatter(all_cvs[mask], all_rates[mask],
                            c=all_force_levels[mask],
                            marker='o' if muscle_type == 'VL' else '^',
                            s=100, alpha=0.7, edgecolors='k',
                            label=muscle_type)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Force Level (% MVC)')

    # Fit quadratic curves for each muscle type
    muscle_types = np.unique(all_muscle_types)
    colors = ['b', 'r']

    for i, muscle_type in enumerate(muscle_types):
        mask = all_muscle_types == muscle_type
        muscle_cvs = all_cvs[mask]
        muscle_rates = all_rates[mask]

        if len(muscle_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(muscle_cvs, muscle_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(muscle_cvs), max(muscle_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{muscle_type} Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(muscle_rates)
            ss_tot = np.sum((muscle_rates - y_mean) ** 2)
            ss_res = np.sum((muscle_rates - poly(muscle_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{muscle_type} R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title('Relationship Between Firing Rate and Discharge Variability\n'
                'Comparing VL and VM Muscles Across All P Folders')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add text with statistics for each muscle type
    stats_text = ""
    for muscle_type in muscle_types:
        mask = all_muscle_types == muscle_type
        muscle_cvs = all_cvs[mask]
        muscle_rates = all_rates[mask]

        stats_text += (f"{muscle_type} (n={len(muscle_cvs)}):\n"
                      f"Mean CV: {np.mean(muscle_cvs):.3f} ± {np.std(muscle_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(muscle_rates):.2f} ± {np.std(muscle_rates):.2f} Hz\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = combined_dir / "P_Combined_CV_vs_Rate_by_Muscle.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved combined CV vs Rate plot to {output_file}")

    # Create a second plot comparing force levels
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate data by force level
    force_levels = np.unique(all_force_levels)
    colors = ['g', 'b', 'r']

    # Fit quadratic curves for each force level
    for i, level in enumerate(force_levels):
        mask = all_force_levels == level
        level_cvs = all_cvs[mask]
        level_rates = all_rates[mask]

        # Create scatter plot for this force level
        scatter = ax.scatter(level_cvs, level_rates,
                            c=[i] * len(level_cvs),  # Same color for all points of this level
                            marker='o',
                            s=100, alpha=0.7, edgecolors='k',
                            label=f'{level}% MVC')

        if len(level_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(level_cvs, level_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(level_cvs), max(level_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{level}% MVC Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(level_rates)
            ss_tot = np.sum((level_rates - y_mean) ** 2)
            ss_res = np.sum((level_rates - poly(level_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{level}% MVC R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title('Relationship Between Firing Rate and Discharge Variability\n'
                'Comparing Force Levels Across All P Folders')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add text with statistics for each force level
    stats_text = ""
    for level in force_levels:
        mask = all_force_levels == level
        level_cvs = all_cvs[mask]
        level_rates = all_rates[mask]

        stats_text += (f"{level}% MVC (n={len(level_cvs)}):\n"
                      f"Mean CV: {np.mean(level_cvs):.3f} ± {np.std(level_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(level_rates):.2f} ± {np.std(level_rates):.2f} Hz\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = combined_dir / "P_Combined_CV_vs_Rate_by_Force.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved combined CV vs Rate by force level plot to {output_file}")

    return fig

def plot_cv_vs_rate_by_force_level(all_cvs, all_rates, all_force_levels, all_muscle_types, title_prefix=""):
    """
    Create a plot with CV on x-axis and firing rate on y-axis, with separate quadratic fits for each force level

    Parameters:
    -----------
    all_cvs : numpy.ndarray
        Array of CV values
    all_rates : numpy.ndarray
        Array of firing rate values
    all_force_levels : numpy.ndarray
        Array of force level values
    all_muscle_types : numpy.ndarray
        Array of muscle type values
    title_prefix : str, optional
        Prefix for the plot title

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create output directory for plots
    combined_dir = Path("P_Combined_Analysis")
    combined_dir.mkdir(exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate data by force level
    force_levels = np.unique(all_force_levels)
    colors = ['g', 'b', 'r', 'c', 'm', 'y']

    # Fit quadratic curves for each force level
    for i, level in enumerate(force_levels):
        mask = all_force_levels == level
        level_cvs = all_cvs[mask]
        level_rates = all_rates[mask]
        level_muscle_types = all_muscle_types[mask]

        # Create scatter plot for this force level with different markers for different muscle types
        for muscle in np.unique(level_muscle_types):
            muscle_mask = level_muscle_types == muscle
            ax.scatter(level_cvs[muscle_mask], level_rates[muscle_mask],
                      c=[colors[i % len(colors)]] * np.sum(muscle_mask),
                      marker='o' if muscle == 'VL' else '^',
                      s=100, alpha=0.7, edgecolors='k',
                      label=f'{level}% MVC - {muscle}')

        if len(level_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(level_cvs, level_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(level_cvs), max(level_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{level}% MVC Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(level_rates)
            ss_tot = np.sum((level_rates - y_mean) ** 2)
            ss_res = np.sum((level_rates - poly(level_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{level}% MVC R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title(f'{title_prefix}Relationship Between Firing Rate and Discharge Variability\n'
                'Comparing Force Levels with Quadratic Fits')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create a custom legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Add text with statistics for each force level
    stats_text = ""
    for level in force_levels:
        mask = all_force_levels == level
        level_cvs = all_cvs[mask]
        level_rates = all_rates[mask]

        stats_text += (f"{level}% MVC (n={len(level_cvs)}):\n"
                      f"Mean CV: {np.mean(level_cvs):.3f} ± {np.std(level_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(level_rates):.2f} ± {np.std(level_rates):.2f} Hz\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = combined_dir / f"P_CV_vs_Rate_by_Force_Level{title_prefix.replace(' ', '_')}.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved CV vs Rate by force level plot to {output_file}")

    return fig

def plot_cv_vs_rate_by_muscle_type(all_cvs, all_rates, all_force_levels, all_muscle_types, title_prefix=""):
    """
    Create a plot with CV on x-axis and firing rate on y-axis, with separate quadratic fits for each muscle type

    Parameters:
    -----------
    all_cvs : numpy.ndarray
        Array of CV values
    all_rates : numpy.ndarray
        Array of firing rate values
    all_force_levels : numpy.ndarray
        Array of force level values
    all_muscle_types : numpy.ndarray
        Array of muscle type values
    title_prefix : str, optional
        Prefix for the plot title

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Create output directory for plots
    combined_dir = Path("P_Combined_Analysis")
    combined_dir.mkdir(exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate data by muscle type
    muscle_types = np.unique(all_muscle_types)
    colors = ['b', 'r']

    # Fit quadratic curves for each muscle type
    for i, muscle_type in enumerate(muscle_types):
        mask = all_muscle_types == muscle_type
        muscle_cvs = all_cvs[mask]
        muscle_rates = all_rates[mask]
        muscle_force_levels = all_force_levels[mask]

        # Create scatter plot for this muscle type with color coding by force level
        scatter = ax.scatter(muscle_cvs, muscle_rates,
                            c=muscle_force_levels,
                            marker='o' if muscle_type == 'VL' else '^',
                            s=100, alpha=0.7, edgecolors='k',
                            label=muscle_type)

        if len(muscle_cvs) > 2:  # Need at least 3 points for quadratic fit
            coeffs = np.polyfit(muscle_cvs, muscle_rates, 2)
            poly = np.poly1d(coeffs)

            # Generate points for the curve
            x_fit = np.linspace(min(muscle_cvs), max(muscle_cvs), 100)
            y_fit = poly(x_fit)

            # Plot the fitted curve
            color = colors[i % len(colors)]
            ax.plot(x_fit, y_fit, f'{color}-', linewidth=2,
                    label=f'{muscle_type} Fit: {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

            # Calculate R-squared
            y_mean = np.mean(muscle_rates)
            ss_tot = np.sum((muscle_rates - y_mean) ** 2)
            ss_res = np.sum((muscle_rates - poly(muscle_cvs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Add R-squared to the plot
            ax.text(0.05, 0.95 - i*0.1, f'{muscle_type} R² = {r_squared:.3f}', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Force Level (% MVC)')

    # Set labels and title
    ax.set_xlabel('Coefficient of Variation (CV) of Interspike Intervals')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title(f'{title_prefix}Relationship Between Firing Rate and Discharge Variability\n'
                'Comparing Muscle Types with Quadratic Fits')

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create a custom legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Add text with statistics for each muscle type
    stats_text = ""
    for muscle_type in muscle_types:
        mask = all_muscle_types == muscle_type
        muscle_cvs = all_cvs[mask]
        muscle_rates = all_rates[mask]

        stats_text += (f"{muscle_type} (n={len(muscle_cvs)}):\n"
                      f"Mean CV: {np.mean(muscle_cvs):.3f} ± {np.std(muscle_cvs):.3f}\n"
                      f"Mean Rate: {np.mean(muscle_rates):.2f} ± {np.std(muscle_rates):.2f} Hz\n\n")

    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    output_file = combined_dir / f"P_CV_vs_Rate_by_Muscle_Type{title_prefix.replace(' ', '_')}.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved CV vs Rate by muscle type plot to {output_file}")

    return fig

def analyze_plateau_data():
    """
    Analyze data from the plateau phases of all P folders
    """
    print("\nAnalyzing plateau phase data from all P folders...")

    # Lists to store data from all motor units
    all_cvs = []
    all_rates = []
    all_muscle_types = []  # To differentiate by muscle type
    all_force_levels = []  # To color-code by force level
    all_subjects = []      # To track which subject (P01 or P02)

    # Process P01_iEMG_Ramps
    p01_path = DATA_PATH / "P01_iEMG_Ramps"
    if p01_path.exists():
        print(f"  Processing {p01_path.name}...")

        # Load ramp definitions (plateau phases)
        p01_ramp_defs = load_ramp_definitions(p01_path)
        if not p01_ramp_defs:
            print(f"  No ramp definitions found in {p01_path}, skipping...")
        else:
            # Process VL data
            vl_files = list_mat_files(p01_path, "VL_")
            for file_path in vl_files:
                try:
                    # Extract force level from filename (e.g., VL_05.mat -> 5)
                    force_level = int(file_path.stem.split('_')[1])

                    # Check if we have ramp definition for this force level
                    if force_level not in p01_ramp_defs:
                        print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                        continue

                    # Get plateau start and stop indices
                    plateau_start, plateau_stop = p01_ramp_defs[force_level]

                    # Load the data
                    data = sio.loadmat(file_path)
                    if 'MUPulses' not in data:
                        continue

                    # Get the MUPulses data
                    mu_pulses = data['MUPulses']

                    # Process each motor unit
                    for i in range(mu_pulses.shape[1]):
                        pulses = mu_pulses[0, i]

                        # Filter pulses to only include those in the plateau phase
                        plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                        # Skip if not enough pulses in plateau
                        if len(plateau_pulses) < 5:
                            print(f"    Skipping P01 VL MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                            continue

                        # Calculate interspike intervals for plateau pulses
                        differences = calculate_discharge_differences(plateau_pulses)

                        # Calculate statistics
                        mean_diff = np.mean(differences)
                        std_diff = np.std(differences)
                        cv = std_diff / mean_diff

                        # Convert to Hz
                        mean_rate = 1000 / mean_diff

                        # Store the data
                        all_cvs.append(cv)
                        all_rates.append(mean_rate)
                        all_muscle_types.append("VL")
                        all_force_levels.append(force_level)
                        all_subjects.append("P01")
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")

            # Process VM data
            vm_files = list_mat_files(p01_path, "VM_")
            for file_path in vm_files:
                try:
                    # Extract force level from filename (e.g., VM_05.mat -> 5)
                    force_level = int(file_path.stem.split('_')[1])

                    # Check if we have ramp definition for this force level
                    if force_level not in p01_ramp_defs:
                        print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                        continue

                    # Get plateau start and stop indices
                    plateau_start, plateau_stop = p01_ramp_defs[force_level]

                    # Load the data
                    data = sio.loadmat(file_path)
                    if 'MUPulses' not in data:
                        continue

                    # Get the MUPulses data
                    mu_pulses = data['MUPulses']

                    # Process each motor unit
                    for i in range(mu_pulses.shape[1]):
                        pulses = mu_pulses[0, i]

                        # Filter pulses to only include those in the plateau phase
                        plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                        # Skip if not enough pulses in plateau
                        if len(plateau_pulses) < 5:
                            print(f"    Skipping P01 VM MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                            continue

                        # Calculate interspike intervals for plateau pulses
                        differences = calculate_discharge_differences(plateau_pulses)

                        # Calculate statistics
                        mean_diff = np.mean(differences)
                        std_diff = np.std(differences)
                        cv = std_diff / mean_diff

                        # Convert to Hz
                        mean_rate = 1000 / mean_diff

                        # Store the data
                        all_cvs.append(cv)
                        all_rates.append(mean_rate)
                        all_muscle_types.append("VM")
                        all_force_levels.append(force_level)
                        all_subjects.append("P01")
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")

    # Process P02_iEMG_Ramps
    p02_path = DATA_PATH / "P02_iEMG_Ramps"
    if p02_path.exists():
        print(f"  Processing {p02_path.name}...")

        # Load ramp definitions (plateau phases)
        p02_ramp_defs = load_ramp_definitions(p02_path)
        if not p02_ramp_defs:
            print(f"  No ramp definitions found in {p02_path}, skipping...")
        else:
            # Process VL data
            vl_files = list_mat_files(p02_path, "VL_")
            for file_path in vl_files:
                try:
                    # Extract force level from filename (e.g., VL_05.mat -> 5)
                    force_level = int(file_path.stem.split('_')[1])

                    # Check if we have ramp definition for this force level
                    if force_level not in p02_ramp_defs:
                        print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                        continue

                    # Get plateau start and stop indices
                    plateau_start, plateau_stop = p02_ramp_defs[force_level]

                    # Load the data
                    data = sio.loadmat(file_path)
                    if 'MUPulses' not in data:
                        continue

                    # Get the MUPulses data
                    mu_pulses = data['MUPulses']

                    # Process each motor unit
                    for i in range(mu_pulses.shape[1]):
                        pulses = mu_pulses[0, i]

                        # Filter pulses to only include those in the plateau phase
                        plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                        # Skip if not enough pulses in plateau
                        if len(plateau_pulses) < 5:
                            print(f"    Skipping P02 VL MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                            continue

                        # Calculate interspike intervals for plateau pulses
                        differences = calculate_discharge_differences(plateau_pulses)

                        # Calculate statistics
                        mean_diff = np.mean(differences)
                        std_diff = np.std(differences)
                        cv = std_diff / mean_diff

                        # Convert to Hz
                        mean_rate = 1000 / mean_diff

                        # Store the data
                        all_cvs.append(cv)
                        all_rates.append(mean_rate)
                        all_muscle_types.append("VL")
                        all_force_levels.append(force_level)
                        all_subjects.append("P02")

                        print(f"    Added P02 VL MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")

            # Process VM data
            vm_files = list_mat_files(p02_path, "VM_")
            for file_path in vm_files:
                try:
                    # Extract force level from filename (e.g., VM_05.mat -> 5)
                    force_level = int(file_path.stem.split('_')[1])

                    # Check if we have ramp definition for this force level
                    if force_level not in p02_ramp_defs:
                        print(f"  No ramp definition for {force_level}% MVC, skipping {file_path}")
                        continue

                    # Get plateau start and stop indices
                    plateau_start, plateau_stop = p02_ramp_defs[force_level]

                    # Load the data
                    data = sio.loadmat(file_path)
                    if 'MUPulses' not in data:
                        continue

                    # Get the MUPulses data
                    mu_pulses = data['MUPulses']

                    # Process each motor unit
                    for i in range(mu_pulses.shape[1]):
                        pulses = mu_pulses[0, i]

                        # Filter pulses to only include those in the plateau phase
                        plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                        # Skip if not enough pulses in plateau
                        if len(plateau_pulses) < 5:
                            print(f"    Skipping P02 VM MU {i+1} in {file_path.name}: Not enough pulses in plateau")
                            continue

                        # Calculate interspike intervals for plateau pulses
                        differences = calculate_discharge_differences(plateau_pulses)

                        # Calculate statistics
                        mean_diff = np.mean(differences)
                        std_diff = np.std(differences)
                        cv = std_diff / mean_diff

                        # Convert to Hz
                        mean_rate = 1000 / mean_diff

                        # Store the data
                        all_cvs.append(cv)
                        all_rates.append(mean_rate)
                        all_muscle_types.append("VM")
                        all_force_levels.append(force_level)
                        all_subjects.append("P02")

                        print(f"    Added P02 VM MU {i+1} from {file_path.name} (plateau): Rate={mean_rate:.2f}Hz, CV={cv:.3f}")
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")

    # Convert to numpy arrays for easier manipulation
    all_cvs = np.array(all_cvs)
    all_rates = np.array(all_rates)
    all_muscle_types = np.array(all_muscle_types)
    all_force_levels = np.array(all_force_levels)
    all_subjects = np.array(all_subjects)

    # Create plots if we have data
    if len(all_cvs) > 0:
        # Plot CV vs Rate by force level
        plot_cv_vs_rate_by_force_level(all_cvs, all_rates, all_force_levels, all_muscle_types, "Plateau Phase - ")

        # Plot CV vs Rate by muscle type
        plot_cv_vs_rate_by_muscle_type(all_cvs, all_rates, all_force_levels, all_muscle_types, "Plateau Phase - ")

        print(f"  Analyzed {len(all_cvs)} motor units from plateau phases")
    else:
        print("  No data found for plateau phases")

def save_plateau_spikes_to_csv(folder_path, muscle_type, ramp_defs):
    """
    Extract spike instants during plateau phases for each motor unit and save to CSV files.
    All plateaus for a given muscle type and force level are combined into a single file.

    Parameters:
    -----------
    folder_path : Path
        Path to the folder containing data files
    muscle_type : str
        Muscle type (VL or VM)
    ramp_defs : dict
        Dictionary with force levels as keys and another dictionary as values.
        The inner dictionary has plateau indices (0, 1, 2, 3) as keys and (start, stop) tuples as values.
    """
    # Create output directory for CSV files
    csv_dir = Path(f"{folder_path.name}_{muscle_type}_Plateau_Spikes_CSV")
    csv_dir.mkdir(exist_ok=True)

    # Get all files for the specified muscle type
    muscle_files = list_mat_files(folder_path, f"{muscle_type}_")

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

            # List to store DataFrames for each plateau
            plateau_dfs = []

            # Metadata for the file
            metadata = [f"# File: {file_path.name}, Muscle: {muscle_type}, Force Level: {force_level}% MVC"]

            # Process each plateau
            for plateau_idx in ramp_defs[force_level].keys():
                # Get plateau start and stop indices
                plateau_start, plateau_stop = ramp_defs[force_level][plateau_idx]

                # Add plateau info to metadata
                metadata.append(f"# Plateau {plateau_idx+1}: {plateau_start/FSAMP:.2f}s - {plateau_stop/FSAMP:.2f}s")

                # Create a list to store rows for this plateau
                plateau_rows = []

                # Extract spikes during plateau phase for each motor unit
                for mu in range(n_units):
                    pulses = mu_pulses[0, mu]

                    # Filter pulses to only include those in the plateau phase
                    plateau_pulses = pulses[(pulses >= plateau_start) & (pulses <= plateau_stop)]

                    # Convert to seconds
                    plateau_pulses_sec = plateau_pulses / FSAMP

                    # Create a row for each spike
                    for spike_time in plateau_pulses_sec:
                        row = {
                            'Spike_Time': spike_time,
                            'MU': mu + 1,
                            'Muscle': muscle_type,
                            'Force_Level': force_level,
                            'Plateau': plateau_idx + 1
                        }
                        plateau_rows.append(row)

                # Skip if no spikes were found in this plateau
                if not plateau_rows:
                    print(f"  No spikes found in plateau {plateau_idx+1} for {file_path.name}, skipping...")
                    continue

                # Create DataFrame for this plateau
                plateau_df = pd.DataFrame(plateau_rows)
                plateau_dfs.append(plateau_df)

                print(f"  Processed plateau {plateau_idx+1} for {file_path.name}: {len(plateau_rows)} spikes")

            # Skip if no plateaus had spikes
            if not plateau_dfs:
                print(f"  No spikes found in any plateau for {file_path.name}, skipping...")
                continue

            # Combine all plateau DataFrames
            combined_df = pd.concat(plateau_dfs, ignore_index=True)

            # Sort by plateau, then MU, then spike time
            combined_df = combined_df.sort_values(by=['Plateau', 'MU', 'Spike_Time'])

            # Add metadata as a comment
            metadata.append("# Values are spike times in seconds")
            metadata_str = "\n".join(metadata) + "\n"

            # Save to CSV
            output_file = csv_dir / f"{folder_path.name}_{muscle_type}_{force_level}_all_plateaus_spikes.csv"

            # Write metadata as comments and then the DataFrame
            with open(output_file, 'w') as f:
                f.write(metadata_str)

            # Append the DataFrame without writing the index
            combined_df.to_csv(output_file, mode='a', index=False)

            print(f"Saved all plateau spikes to {output_file}")

            # NOVO: Plota o sinal de força com CV dos plateaus e metadados
            if 'ref_signal' in data:
                ref_signal = data['ref_signal'].flatten()
                plot_force_with_plateau_cv(file_path, ref_signal, ramp_defs[force_level], force_level, csv_dir, FSAMP, metadata)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    return None

def plot_force_with_plateau_cv(file_path, ref_signal, ramp_defs, force_level, output_dir, FSAMP=10240, metadata=None):
    """
    Plot the force signal with highlighted plateaus, showing the CV of each plateau and including CSV metadata in the figure.

    Parameters:
    -----------
    file_path : Path
        Path to the .mat file
    ref_signal : numpy.ndarray
        Force signal data
    ramp_defs : dict
        Dictionary with plateau indices as keys and (start, stop) tuples as values
    force_level : int
        Force level (% MVC)
    output_dir : Path
        Directory to save the output figure
    FSAMP : int, optional
        Sampling frequency (default: 10240)
    metadata : list, optional
        List of metadata strings to include in the figure

    Returns:
    --------
    None
    """
    time_ref = np.arange(len(ref_signal)) / FSAMP
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_ref, ref_signal, 'k-', label='Force (% MVC)')

    plateau_colors = ['g', 'b', 'r', 'c', 'm', 'y']
    for plateau_idx, (start_idx, stop_idx) in ramp_defs.items():
        start_time = start_idx / FSAMP
        stop_time = stop_idx / FSAMP
        color = plateau_colors[plateau_idx % len(plateau_colors)]
        plateau_signal = ref_signal[start_idx:stop_idx+1]
        if len(plateau_signal) > 1:
            mean = np.mean(plateau_signal)
            std = np.std(plateau_signal)
            cv = std / mean if mean != 0 else np.nan
        else:
            cv = np.nan
        ax.axvspan(start_time, stop_time, color=color, alpha=0.2)

        # Make sure CV is prominently displayed on the plot
        ax.text((start_time+stop_time)/2, np.max(ref_signal)*0.95, f'CV: {cv:.3f}',
                color=color, ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axvline(x=start_time, color=color, linestyle='--', alpha=0.7)
        ax.axvline(x=stop_time, color=color, linestyle=':', alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (% MVC)')

    # Include CSV metadata at the top of the figure
    if metadata is not None:
        metadata_str = "\n".join(metadata)
        fig.text(0.5, 0.99, metadata_str, ha='center', va='top', fontsize=10, wrap=True, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f'Force Signal with Plateau CVs - {file_path.name} ({force_level}% MVC)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout(rect=[0,0,1,0.93])
    output_file = output_dir / f"{file_path.stem}_force_plateau_CV.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved force+CV plot to {output_file}")

    return None

def compute_isi_statistics(folder_path, muscle_type, ramp_defs):
    """
    Compute ISI (Inter-Spike Interval) mean and standard deviation for each motor unit,
    organized by muscle type and force level.

    Parameters:
    -----------
    folder_path : Path
        Path to the folder containing data files
    muscle_type : str
        Muscle type (VL or VM)
    ramp_defs : dict
        Dictionary with force levels as keys and another dictionary as values.
        The inner dictionary has plateau indices (0, 1, 2, 3) as keys and (start, stop) tuples as values.

    Returns:
    --------
    df_stats : pandas.DataFrame
        DataFrame containing ISI statistics for each motor unit
    """
    # List to store statistics for each motor unit
    stats_rows = []

    # Get all files for the specified muscle type
    muscle_files = list_mat_files(folder_path, f"{muscle_type}_")

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
                        'Folder': folder_path.name,
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

    # Create DataFrame from all rows
    df_stats = pd.DataFrame(stats_rows)

    # Sort by folder, muscle, force level, plateau, and MU
    if not df_stats.empty:
        df_stats = df_stats.sort_values(by=['Folder', 'Muscle', 'Force_Level', 'Plateau', 'MU'])

    return df_stats

