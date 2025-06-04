#!/usr/bin/env python3
"""
PyNN simulation with two populations of spikeGammaRate neurons.

This script creates two populations of spikeGammaRate neurons, each with one neuron,
and generates histograms of their ISIs (Inter-Spike Intervals) as well as a combined
histogram of both populations' ISIs.

Each population has a different order (alpha) and rate (beta) parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyNN.utility import get_simulator
import pyNN.neuron as sim
import os
import sys

# Import the custom Classes module
sys.path.append("./src")
import src.Classes as Classes

# Create output directory for figures
OUTPUT_DIR = "gamma_simulation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_simulation(sim_time=20000.0, dt=0.1):
    """
    Run a PyNN simulation with two populations of spikeGammaRate neurons.

    Parameters:
    -----------
    sim_time : float
        Simulation time in milliseconds
    dt : float
        Time step in milliseconds

    Returns:
    --------
    spike_trains : dict
        Dictionary containing spike trains for both populations
    """
    # Set up the simulator
    sim.setup(timestep=dt, min_delay=dt)

    # Create three populations of spikeGammaRate neurons with different parameters
    # Population 1: Higher order (alpha), lower rate (beta)
    pop1 = sim.Population(1, Classes.SpikeSourceGammaStart(
        alpha=5.0,     # Order parameter (higher = more regular)
        beta=100,        # Rate parameter in Hz
        start=0.0,     # Start time in ms
        duration=sim_time  # Duration in ms
    ))

    # Population 2: Lower order (alpha), higher rate (beta)
    pop2 = sim.Population(1, Classes.SpikeSourceGammaStart(
        alpha=5.0,     # Order parameter (lower = more variable)
        beta=100,        # Rate parameter in Hz
        start=0.0,     # Start time in ms
        duration=sim_time  # Duration in ms
    ))

    # Population 3: Medium order, medium rate (will not be included in combined histogram)
    pop3 = sim.Population(1, Classes.SpikeSourceGammaStart(
        alpha=10,     # Order parameter (medium variability)
        beta=100,        # Rate parameter in Hz
        start=0.0,     # Start time in ms
        duration=sim_time  # Duration in ms
    ))

    # Record spikes from all populations
    pop1.record('spikes')
    pop2.record('spikes')
    pop3.record('spikes')

    # Run the simulation
    sim.run(sim_time)

    # Get spike data
    spike_data_pop1 = pop1.get_data().segments[0].spiketrains
    spike_data_pop2 = pop2.get_data().segments[0].spiketrains
    spike_data_pop3 = pop3.get_data().segments[0].spiketrains

    print(f"Population 1 spike count: {len(spike_data_pop1[0])}")
    print(f"Population 2 spike count: {len(spike_data_pop2[0])}")
    print(f"Population 3 spike count: {len(spike_data_pop3[0])}")

    # Convert spike trains to numpy arrays
    spike_trains = {
        'pop1': np.array(spike_data_pop1[0]),
        'pop2': np.array(spike_data_pop2[0]),
        'pop3': np.array(spike_data_pop3[0])
    }

    # Print first few spikes for debugging
    if len(spike_trains['pop1']) > 0:
        print(f"First 5 spikes from population 1: {spike_trains['pop1'][:5]}")
    if len(spike_trains['pop2']) > 0:
        print(f"First 5 spikes from population 2: {spike_trains['pop2'][:5]}")
    if len(spike_trains['pop3']) > 0:
        print(f"First 5 spikes from population 3: {spike_trains['pop3'][:5]}")

    # End the simulation
    sim.end()

    return spike_trains

def calculate_isis(spike_train):
    """
    Calculate Inter-Spike Intervals (ISIs) from a spike train.

    Parameters:
    -----------
    spike_train : numpy.ndarray
        Array of spike times

    Returns:
    --------
    isis : numpy.ndarray
        Array of inter-spike intervals
    """
    print(f"Calculating ISIs for spike train with {len(spike_train)} spikes")

    if len(spike_train) < 2:
        print("Not enough spikes to calculate ISIs (need at least 2)")
        return np.array([])

    # Calculate differences between consecutive spikes
    isis = np.diff(spike_train)

    print(f"Calculated {len(isis)} ISIs")
    if len(isis) > 0:
        print(f"ISI range: {np.min(isis):.2f} to {np.max(isis):.2f} ms")
        print(f"Mean ISI: {np.mean(isis):.2f} ms (approx. {1000/np.mean(isis):.2f} Hz)")

    return isis

def plot_isi_histograms(spike_trains, bins=50, save_path=None):
    """
    Plot histograms of ISIs for all populations and a combined histogram of populations 1 and 2.

    Parameters:
    -----------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    bins : int
        Number of bins for the histograms
    save_path : str, optional
        Path to save the figure
    """
    # Calculate ISIs for all populations
    isis_pop1 = calculate_isis(spike_trains['pop1'])
    isis_pop2 = calculate_isis(spike_trains['pop2'])
    isis_pop3 = calculate_isis(spike_trains['pop3'])

    # Combine spike times from populations 1 and 2 only (excluding population 3)
    print("Combining spike times from populations 1 and 2 only (excluding population 3)...")
    combined_spikes = np.concatenate([spike_trains['pop1'], spike_trains['pop2']])

    # Sort the combined spike times chronologically
    combined_spikes.sort()

    print(f"Combined spike train has {len(combined_spikes)} spikes")

    # Calculate ISIs from the combined spike train
    combined_isis = calculate_isis(combined_spikes)

    # Create figure with 5 subplots (3 populations + 2 combined methods)
    fig, axes = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

    # Plot histogram for population 1
    axes[0].hist(isis_pop1, bins=bins, alpha=0.7, color='blue')
    axes[0].set_title('Population 1 (alpha=5.0, beta=5 Hz) ISI Histogram')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Add statistics for population 1
    if len(isis_pop1) > 0:
        mean_isi_pop1 = np.mean(isis_pop1)
        std_isi_pop1 = np.std(isis_pop1)
        cv_isi_pop1 = std_isi_pop1 / mean_isi_pop1 if mean_isi_pop1 > 0 else 0

        stats_text_pop1 = (f"Mean ISI: {mean_isi_pop1:.2f} ms\n"
                          f"Std Dev: {std_isi_pop1:.2f} ms\n"
                          f"CV: {cv_isi_pop1:.3f}\n"
                          f"Firing Rate: {1000/mean_isi_pop1:.2f} Hz\n"
                          f"Number of Spikes: {len(spike_trains['pop1'])}")

        axes[0].text(0.95, 0.95, stats_text_pop1, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot histogram for population 2
    axes[1].hist(isis_pop2, bins=bins, alpha=0.7, color='red')
    axes[1].set_title('Population 2 (alpha=2.0, beta=4 Hz) ISI Histogram')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Add statistics for population 2
    if len(isis_pop2) > 0:
        mean_isi_pop2 = np.mean(isis_pop2)
        std_isi_pop2 = np.std(isis_pop2)
        cv_isi_pop2 = std_isi_pop2 / mean_isi_pop2 if mean_isi_pop2 > 0 else 0

        stats_text_pop2 = (f"Mean ISI: {mean_isi_pop2:.2f} ms\n"
                          f"Std Dev: {std_isi_pop2:.2f} ms\n"
                          f"CV: {cv_isi_pop2:.3f}\n"
                          f"Firing Rate: {1000/mean_isi_pop2:.2f} Hz\n"
                          f"Number of Spikes: {len(spike_trains['pop2'])}")

        axes[1].text(0.95, 0.95, stats_text_pop2, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot histogram for population 3
    axes[2].hist(isis_pop3, bins=bins, alpha=0.7, color='purple')
    axes[2].set_title('Population 3 (alpha=3.0, beta=7 Hz) ISI Histogram')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # Add statistics for population 3
    if len(isis_pop3) > 0:
        mean_isi_pop3 = np.mean(isis_pop3)
        std_isi_pop3 = np.std(isis_pop3)
        cv_isi_pop3 = std_isi_pop3 / mean_isi_pop3 if mean_isi_pop3 > 0 else 0

        stats_text_pop3 = (f"Mean ISI: {mean_isi_pop3:.2f} ms\n"
                          f"Std Dev: {std_isi_pop3:.2f} ms\n"
                          f"CV: {cv_isi_pop3:.3f}\n"
                          f"Firing Rate: {1000/mean_isi_pop3:.2f} Hz\n"
                          f"Number of Spikes: {len(spike_trains['pop3'])}\n"
                          f"(Not included in combined histogram)")

        axes[2].text(0.95, 0.95, stats_text_pop3, transform=axes[2].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot combined histogram (populations 1 and 2 only)
    axes[3].hist(combined_isis, bins=bins, alpha=0.7, color='green')
    axes[3].set_title('ISI Histogram from Combined Spike Train (Populations 1 and 2 only)')
    axes[3].set_xlabel('Inter-Spike Interval (ms)')
    axes[3].set_ylabel('Count')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    # Add statistics for combined ISIs
    if len(combined_isis) > 0:
        mean_isi_combined = np.mean(combined_isis)
        std_isi_combined = np.std(combined_isis)
        cv_isi_combined = std_isi_combined / mean_isi_combined if mean_isi_combined > 0 else 0

        stats_text_combined = (f"Mean ISI: {mean_isi_combined:.2f} ms\n"
                              f"Std Dev: {std_isi_combined:.2f} ms\n"
                              f"CV: {cv_isi_combined:.3f}\n"
                              f"Firing Rate: {1000/mean_isi_combined:.2f} Hz\n"
                              f"Number of Spikes: {len(combined_spikes)}\n"
                              f"(Combined and chronologically sorted)")

        axes[3].text(0.95, 0.95, stats_text_combined, transform=axes[3].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot histogram of concatenated ISIs (Method 2: Concatenate ISIs directly)
    # Simply concatenate the ISIs from populations 1 and 2 without combining spike trains first
    concatenated_isis = np.concatenate([isis_pop1, isis_pop2])

    print(f"Concatenated ISIs method: {len(concatenated_isis)} ISIs")

    axes[4].hist(concatenated_isis, bins=bins, alpha=0.7, color='orange')
    axes[4].set_title('ISI Histogram from Concatenated ISIs (Populations 1 and 2 only)')
    axes[4].set_xlabel('Inter-Spike Interval (ms)')
    axes[4].set_ylabel('Count')
    axes[4].grid(True, linestyle='--', alpha=0.7)

    # Add statistics for concatenated ISIs
    if len(concatenated_isis) > 0:
        mean_isi_concat = np.mean(concatenated_isis)
        std_isi_concat = np.std(concatenated_isis)
        cv_isi_concat = std_isi_concat / mean_isi_concat if mean_isi_concat > 0 else 0

        stats_text_concat = (f"Mean ISI: {mean_isi_concat:.2f} ms\n"
                            f"Std Dev: {std_isi_concat:.2f} ms\n"
                            f"CV: {cv_isi_concat:.3f}\n"
                            f"Firing Rate: {1000/mean_isi_concat:.2f} Hz\n"
                            f"Number of ISIs: {len(concatenated_isis)}\n"
                            f"(Direct concatenation of ISIs)")

        axes[4].text(0.95, 0.95, stats_text_concat, transform=axes[4].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    return fig

def main():
    """
    Main function to run the simulation and generate histograms.
    """
    print("Starting PyNN simulation with two populations of spikeGammaRate neurons...")

    # Run the simulation
    sim_time = 20000.0  # 20 seconds
    spike_trains = run_simulation(sim_time=sim_time)

    # Plot ISI histograms
    output_file = os.path.join(OUTPUT_DIR, "gamma_rate_isi_histograms.png")
    plot_isi_histograms(spike_trains, bins=50, save_path=output_file)

    # Print summary
    print("\nSimulation Summary:")
    print(f"Simulation time: {sim_time} ms")
    print(f"Population 1 (alpha=5.0, beta=5 Hz) spikes: {len(spike_trains['pop1'])}")
    print(f"Population 2 (alpha=2.0, beta=4 Hz) spikes: {len(spike_trains['pop2'])}")
    print(f"Population 3 (alpha=3.0, beta=7 Hz) spikes: {len(spike_trains['pop3'])}")
    print(f"Total spikes in combined histogram (pop1 + pop2): {len(spike_trains['pop1']) + len(spike_trains['pop2'])}")
    print(f"Total spikes overall: {len(spike_trains['pop1']) + len(spike_trains['pop2']) + len(spike_trains['pop3'])}")
    print("\nHistogram Information:")
    print("1. Population 1 ISI Histogram")
    print("2. Population 2 ISI Histogram")
    print("3. Population 3 ISI Histogram (not included in combined histograms)")
    print("4. Combined Spike Train ISI Histogram (Method 1: Combine spikes first, then calculate ISIs)")
    print("5. Concatenated ISIs Histogram (Method 2: Calculate ISIs separately, then concatenate)")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
