#!/usr/bin/env python3
"""
Script to generate spikes for 3 populations of neurons using PyNN.

Each population has 2 neurons firing at an average rate of 5 Hz.
The script generates histograms for each population's spikes.
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Set matplotlib backend to QtAgg for interactive display
import matplotlib.pyplot as plt
import os
import sys

# Import PyNN
import pyNN.neuron as sim

# Import custom Classes module
sys.path.append("./src")
import src.Classes as Classes

# Create output directory for figures
OUTPUT_DIR = "spike_simulation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_simulation(sim_time=20000.0, dt=0.1):
    """
    Run a PyNN simulation with three populations of neurons.

    Each population has 2 neurons firing at an average rate of 5 Hz.

    Parameters:
    -----------
    sim_time : float
        Simulation time in milliseconds
    dt : float
        Time step in milliseconds

    Returns:
    --------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    """
    # Set up the simulator
    sim.setup(timestep=dt, min_delay=dt)

    # Create three populations of neurons, each with 2 neurons
    # All populations have the same average firing rate of 5 Hz (beta=5)
    # but different alpha values to create different firing patterns

    # Population 1: Higher order (alpha=3)
    pop1 = sim.Population(2, Classes.SpikeSourceGammaStart(
        alpha=3.0,     # Order parameter (higher = more regular)
        beta=5.0,      # Rate parameter in Hz (5 Hz)
        start=0.0,     # Start time in ms
        duration=sim_time  # Duration in ms
    ))

    # Population 2: Medium order (alpha=2)
    pop2 = sim.Population(2, Classes.SpikeSourceGammaStart(
        alpha=2.0,     # Order parameter (medium variability)
        beta=5.0,      # Rate parameter in Hz (5 Hz)
        start=0.0,     # Start time in ms
        duration=sim_time  # Duration in ms
    ))

    # Population 3: Lower order (alpha=1)
    pop3 = sim.Population(2, Classes.SpikeSourceGammaStart(
        alpha=1.0,     # Order parameter (lower = more variable)
        beta=5.0,      # Rate parameter in Hz (5 Hz)
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

    # Convert spike trains to numpy arrays and organize by population
    spike_trains = {
        'pop1': [np.array(spike_data_pop1[0]), np.array(spike_data_pop1[1])],
        'pop2': [np.array(spike_data_pop2[0]), np.array(spike_data_pop2[1])],
        'pop3': [np.array(spike_data_pop3[0]), np.array(spike_data_pop3[1])]
    }

    # Print spike counts for each population
    for pop_name, pop_spikes in spike_trains.items():
        total_spikes = sum(len(neuron_spikes) for neuron_spikes in pop_spikes)
        print(f"{pop_name} total spike count: {total_spikes}")
        for i, neuron_spikes in enumerate(pop_spikes):
            print(f"  Neuron {i} spike count: {len(neuron_spikes)}")

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
    if len(spike_train) < 2:
        return np.array([])

    # Calculate differences between consecutive spikes
    isis = np.diff(spike_train)

    return isis

def plot_population_histograms(spike_trains, bins=50, save_path=None):
    """
    Plot histograms of ISIs for each population.

    Parameters:
    -----------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    bins : int
        Number of bins for the histograms
    save_path : str, optional
        Path to save the figure
    """
    # Create figure with 3 subplots (one for each population)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Colors for each neuron in the populations
    colors = ['blue', 'cyan', 'red', 'salmon', 'green', 'lightgreen']

    # Population names and their alpha values for titles
    pop_info = [
        ('pop1', 3.0),
        ('pop2', 2.0),
        ('pop3', 1.0)
    ]

    # Process each population
    for i, (pop_name, alpha) in enumerate(pop_info):
        # Get spike trains for both neurons in this population
        neuron_spikes = spike_trains[pop_name]

        # Calculate ISIs for each neuron
        neuron_isis = [calculate_isis(spikes) for spikes in neuron_spikes]

        # Combine ISIs from both neurons in this population
        combined_isis = np.concatenate(neuron_isis)

        # Plot histogram for this population
        axes[i].hist(combined_isis, bins=bins, alpha=0.7, color=colors[i*2])
        axes[i].set_title(f'Population {i+1} (alpha={alpha}, beta=5 Hz) ISI Histogram')
        axes[i].set_ylabel('Count')
        axes[i].grid(True, linestyle='--', alpha=0.7)

        # Add statistics for this population
        if len(combined_isis) > 0:
            mean_isi = np.mean(combined_isis)
            std_isi = np.std(combined_isis)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else 0

            total_spikes = sum(len(spikes) for spikes in neuron_spikes)

            stats_text = (f"Mean ISI: {mean_isi:.2f} ms\n"
                         f"Std Dev: {std_isi:.2f} ms\n"
                         f"CV: {cv_isi:.3f}\n"
                         f"Firing Rate: {1000/mean_isi:.2f} Hz\n"
                         f"Number of Spikes: {total_spikes}")

            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Set x-label for the bottom subplot
    axes[2].set_xlabel('Inter-Spike Interval (ms)')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Population histograms saved to {save_path}")

    return fig

def plot_individual_neuron_histograms(spike_trains, bins=50, save_path=None):
    """
    Plot histograms of ISIs for each individual neuron.

    Parameters:
    -----------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    bins : int
        Number of bins for the histograms
    save_path : str, optional
        Path to save the figure
    """
    # Create figure with 6 subplots (one for each neuron)
    fig, axes = plt.subplots(6, 1, figsize=(10, 20), sharex=True)

    # Colors for each neuron
    colors = ['blue', 'cyan', 'red', 'salmon', 'green', 'lightgreen']

    # Population names for titles
    pop_names = ['pop1', 'pop2', 'pop3']

    # Process each neuron
    neuron_idx = 0
    for pop_idx, pop_name in enumerate(pop_names):
        # Get spike trains for both neurons in this population
        neuron_spikes = spike_trains[pop_name]

        # Process each neuron in this population
        for i, spikes in enumerate(neuron_spikes):
            # Calculate ISIs for this neuron
            isis = calculate_isis(spikes)

            # Plot histogram for this neuron
            axes[neuron_idx].hist(isis, bins=bins, alpha=0.7, color=colors[neuron_idx])
            axes[neuron_idx].set_title(f'Population {pop_idx+1}, Neuron {i+1} ISI Histogram')
            axes[neuron_idx].set_ylabel('Count')
            axes[neuron_idx].grid(True, linestyle='--', alpha=0.7)

            # Add statistics for this neuron
            if len(isis) > 0:
                mean_isi = np.mean(isis)
                std_isi = np.std(isis)
                cv_isi = std_isi / mean_isi if mean_isi > 0 else 0

                stats_text = (f"Mean ISI: {mean_isi:.2f} ms\n"
                             f"Std Dev: {std_isi:.2f} ms\n"
                             f"CV: {cv_isi:.3f}\n"
                             f"Firing Rate: {1000/mean_isi:.2f} Hz\n"
                             f"Number of Spikes: {len(spikes)}")

                axes[neuron_idx].text(0.95, 0.95, stats_text, transform=axes[neuron_idx].transAxes,
                                    verticalalignment='top', horizontalalignment='right',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            neuron_idx += 1

    # Set x-label for the bottom subplot
    axes[5].set_xlabel('Inter-Spike Interval (ms)')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Individual neuron histograms saved to {save_path}")

    return fig

def plot_spike_raster(spike_trains, sim_time, save_path=None):
    """
    Create a raster plot showing spike times for all neurons.

    Parameters:
    -----------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    sim_time : float
        Total simulation time in milliseconds
    save_path : str, optional
        Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for each population
    pop_colors = ['blue', 'red', 'green']

    # Population names
    pop_names = ['pop1', 'pop2', 'pop3']

    # Track neuron index for y-axis
    neuron_idx = 0

    # Plot spikes for each neuron
    for pop_idx, pop_name in enumerate(pop_names):
        # Get spike trains for both neurons in this population
        neuron_spikes = spike_trains[pop_name]

        # Process each neuron in this population
        for i, spikes in enumerate(neuron_spikes):
            # Plot each spike as a vertical line
            for spike_time in spikes:
                ax.vlines(spike_time, neuron_idx + 0.5, neuron_idx + 1.5,
                         color=pop_colors[pop_idx], alpha=0.7)

            # Add label for this neuron
            ax.text(-500, neuron_idx + 1, f'Pop {pop_idx+1}, N{i+1}',
                   fontsize=10, ha='right', va='center')

            neuron_idx += 1

    # Set plot limits and labels
    ax.set_xlim(-200, sim_time + 200)
    ax.set_ylim(0.5, neuron_idx + 0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_title('Spike Raster Plot')

    # Add legend for populations
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'Population {i+1}')
                      for i, color in enumerate(pop_colors)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Spike raster plot saved to {save_path}")

    return fig

def plot_firing_rates(spike_trains, sim_time, bin_size=500, save_path=None):
    """
    Plot firing rates over time for each neuron and population.

    Parameters:
    -----------
    spike_trains : dict
        Dictionary containing spike trains for all populations
    sim_time : float
        Total simulation time in milliseconds
    bin_size : float
        Size of time bins in milliseconds for calculating firing rates
    save_path : str, optional
        Path to save the figure
    """
    # Create figure with 2 subplots (individual neurons and population averages)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Create time bins
    bins = np.arange(0, sim_time + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2

    # Colors for each neuron and population
    neuron_colors = ['blue', 'cyan', 'red', 'salmon', 'green', 'lightgreen']
    pop_colors = ['blue', 'red', 'green']

    # Population names for legend
    pop_names = ['pop1', 'pop2', 'pop3']

    # Dictionary to store population firing rates
    pop_rates = {pop_name: np.zeros(len(bins)-1) for pop_name in pop_names}

    # Plot firing rates for individual neurons
    neuron_idx = 0
    for pop_idx, pop_name in enumerate(pop_names):
        # Get spike trains for both neurons in this population
        neuron_spikes = spike_trains[pop_name]

        # Process each neuron in this population
        for i, spikes in enumerate(neuron_spikes):
            # Calculate histogram of spikes
            hist, _ = np.histogram(spikes, bins=bins)

            # Convert to firing rate in Hz (spikes per second)
            firing_rate = hist * (1000 / bin_size)

            # Plot firing rate for this neuron
            axes[0].plot(bin_centers, firing_rate, color=neuron_colors[neuron_idx],
                        alpha=0.7, linewidth=1.5,
                        label=f'Pop {pop_idx+1}, Neuron {i+1}')

            # Add to population average
            pop_rates[pop_name] += firing_rate

            neuron_idx += 1

    # Set labels for individual neurons plot
    axes[0].set_title('Firing Rates of Individual Neurons')
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper right')

    # Plot population averages
    for pop_idx, pop_name in enumerate(pop_names):
        # Calculate average firing rate for this population (divide by number of neurons = 2)
        avg_rate = pop_rates[pop_name] / 2

        # Plot average firing rate for this population
        axes[1].plot(bin_centers, avg_rate, color=pop_colors[pop_idx],
                    linewidth=2.5, label=f'Population {pop_idx+1}')

    # Set labels for population averages plot
    axes[1].set_title('Average Firing Rates by Population')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper right')

    # Add horizontal line at 5 Hz (target rate)
    axes[0].axhline(y=5, color='black', linestyle='--', alpha=0.5, label='Target Rate (5 Hz)')
    axes[1].axhline(y=5, color='black', linestyle='--', alpha=0.5, label='Target Rate (5 Hz)')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Firing rate plot saved to {save_path}")

    return fig

def main():
    """
    Main function to run the simulation and generate histograms.
    """
    print("Starting PyNN simulation with three populations of neurons...")

    # Run the simulation
    sim_time = 20000.0  # 20 seconds
    spike_trains = run_simulation(sim_time=sim_time)

    # Plot population ISI histograms
    pop_output_file = os.path.join(OUTPUT_DIR, "population_isi_histograms.png")
    plot_population_histograms(spike_trains, bins=50, save_path=pop_output_file)

    # Plot individual neuron ISI histograms
    neuron_output_file = os.path.join(OUTPUT_DIR, "individual_neuron_isi_histograms.png")
    plot_individual_neuron_histograms(spike_trains, bins=50, save_path=neuron_output_file)

    # Plot spike raster
    raster_output_file = os.path.join(OUTPUT_DIR, "spike_raster.png")
    plot_spike_raster(spike_trains, sim_time, save_path=raster_output_file)

    # Plot firing rates over time
    firing_rate_output_file = os.path.join(OUTPUT_DIR, "firing_rates.png")
    plot_firing_rates(spike_trains, sim_time, bin_size=500, save_path=firing_rate_output_file)

    # Show the plots
    plt.show()

    # Print summary
    print("\nSimulation Summary:")
    print(f"Simulation time: {sim_time} ms")
    for pop_name in spike_trains:
        total_spikes = sum(len(neuron_spikes) for neuron_spikes in spike_trains[pop_name])
        print(f"{pop_name} total spikes: {total_spikes}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
