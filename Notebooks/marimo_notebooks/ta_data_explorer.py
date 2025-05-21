import sys

sys.path.append("./../")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from pathlib import Path
import os
import csv

# set backend to Agg (non-GUI backend for saving figures)
plt.switch_backend("Agg")

# Constants from README
FSAMP = 10240  # Sampling frequency for all datasets

# Data folder path
DATA_PATH = Path("data")

# Load plateau information from CSV file
def load_plateau_info():
    """
    Load plateau information from CSV file

    Returns:
    --------
    plateau_info : dict
        Dictionary with filenames as keys and plateau information as values
    """
    plateau_info = {}
    csv_path = Path("Notebooks/marimo_notebooks/ta_plateaus.csv")

    if not csv_path.exists():
        print(f"Warning: Plateau info file {csv_path} not found")
        return plateau_info

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            plateau_info[filename] = {
                'plateau1': (float(row['plateau1_start']), float(row['plateau1_end'])),
                'plateau2': (float(row['plateau2_start']), float(row['plateau2_end']))
            }

    print(f"Loaded plateau information for {len(plateau_info)} files")
    return plateau_info

# Global variable to store plateau information
PLATEAU_INFO = load_plateau_info()

def list_mat_files(folder_path):
    """List all .mat files in a given folder"""
    mat_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.mat']
    return mat_files

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
                    ax_ref.set_ylabel('Force (% MVC)')  # Note: TA force is in % MVC
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

def export_plateau_spikes_csv():
    """
    Exporta arquivos CSV com os spikes de cada unidade motora em cada plateau,
    usando nomes de colunas definidos explicitamente e adicionando metadata no início.
    """
    ta_path = DATA_PATH / "TA"
    output_dir = Path("TA_plateau_spikes_csv")
    output_dir.mkdir(exist_ok=True)

    # Defina explicitamente os nomes das colunas conforme o padrão desejado
    fieldnames = ["Spike_Time", "MU", "Muscle", "Force_Level", "Plateau"]

    mat_files = list_mat_files(ta_path)

    for mat_file in mat_files:
        try:
            data = sio.loadmat(mat_file)
            metadata = [f"# File: {mat_file.name}",
                        f"# Muscle: TA"]
            force_level = mat_file.name.split('_')[1][1:]
            metadata.append(f"# Force Level: {force_level}% MVC")
            metadata.append(f"# Values are spike times in seconds")
            metadata.append(f"#")
            metadata.append(f"#")
            if 'MUPulses' not in data:
                continue

            mu_pulses = data['MUPulses']
            n_units = mu_pulses.shape[1]
            # Verifica plateaus no CSV
            if mat_file.name not in PLATEAU_INFO:
                print(f"Sem info de plateau para {mat_file.name}, pulando...")
                continue

            plateau_info = PLATEAU_INFO[mat_file.name]
            rows = []
            
            for mu_idx in range(n_units):
                pulses = mu_pulses[0, mu_idx]
                for plateau_name, (start_sec, end_sec) in plateau_info.items():
                    start_idx = int(start_sec * FSAMP)
                    end_idx = int(end_sec * FSAMP)
                    plateau_pulses = pulses[(pulses >= start_idx) & (pulses <= end_idx)]

                    for spike in plateau_pulses:
                        spike_time_sec = spike / FSAMP
                        row = {
                            "Spike_Time": spike_time_sec,
                            "MU": mu_idx + 1,
                            "Muscle": "TA",
                            "Force_Level": force_level,
                            "Plateau": int(plateau_name[-1]),                            
                        }
                        rows.append(row)

            output_csv = output_dir / f"{mat_file.stem}_all_plateaus_spikes.csv"
            with open(output_csv, "w", newline="") as f:
                # Escreve o metadata antes do cabeçalho
                for line in metadata:
                    f.write(line + "\n")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Exported: {output_csv}")

        except Exception as e:
            print(f"Procesing error {mat_file.name}: {e}")

if __name__ == "__main__":
    print("TA Muscle Data Analyzer")
    print("=======================")
    print("\nEste programa analisa dados do TA e gera gráficos.")

    # Cria CSVs de spikes por plateau
    print("\nExportando CSVs de spikes por plateau...")
    export_plateau_spikes_csv()

    # Remova ou comente a linha abaixo, pois a função não existe:
    # print("\nCriando gráfico CV vs Rate com ajustes quadráticos separados por nível de força (dados completos)")
    # plot_cv_vs_rate_whole_data()