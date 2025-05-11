import sys

sys.path.append("./../")
import pyNN.neuron as sim
import numpy as np
import matplotlib.pyplot as plt
from pyNN.neuron.morphology import (
    uniform,
    centre,
)
from pyNN.parameters import IonicSpecies
import src.Classes as Classes
import src.functions as funçoes


def create_motor_neuron_pool(n_neurons):
    """Create a pool of motor neurons with specified parameters"""
    somas = funçoes.create_somas(n_neurons)
    dends = funçoes.create_dends(n_neurons, somas)

    cell_type = Classes.cell_class(
        morphology=funçoes.soma_dend(somas, dends),
        cm=1,  # mF / cm**2
        Ra=0.070,  # ohm.mm
        ionic_species={
            "na": IonicSpecies("na", reversal_potential=50),
            "ks": IonicSpecies("ks", reversal_potential=-80),
            "kf": IonicSpecies("kf", reversal_potential=-80),
        },
        pas_soma={"conductance_density": uniform("soma", 7e-4), "e_rev": -70},
        pas_dend={"conductance_density": uniform("dendrite", 7e-4), "e_rev": -70},
        na={
            "conductance_density": uniform("soma", 10),
            "vt": list(np.linspace(-57.65, -53, n_neurons)),
        },
        kf={
            "conductance_density": uniform("soma", 1),
            "vt": list(np.linspace(-57.65, -53, n_neurons)),
        },
        ks={
            "conductance_density": uniform("soma", 0.5),
            "vt": list(np.linspace(-57.65, -53, n_neurons)),
        },
        syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
    )
    return cell_type


def run_simulation(current_matrix, neurons_per_pool=100):
    """
    Run simulation with custom current inputs

    Parameters:
    -----------
    current_matrix : numpy.ndarray
        Matrix of shape (n_pools, t_points) containing current values
        Each row represents the current for one pool
    neurons_per_pool : int
        Number of neurons in each pool
    """
    # Validate input
    if not isinstance(current_matrix, np.ndarray):
        raise TypeError("current_matrix must be a numpy array")
    if current_matrix.ndim != 2:
        raise ValueError("current_matrix must be 2-dimensional")

    # Setup simulation with proper parameters
    timestep = 0.05  # ms
    min_delay = 0.1  # ms
    sim.setup(timestep=timestep, min_delay=min_delay)

    try:
        # Create motor neuron pools
        n_pools = len(current_matrix)
        pools = []
        for i in range(n_pools):
            cell_type = create_motor_neuron_pool(neurons_per_pool)
            pool = sim.Population(
                neurons_per_pool,
                cell_type,
                initial_values={"v": -70},  # Simplified initialization
            )
            pools.append(pool)

        # Inject currents into each pool - one current per pool
        for i, pool in enumerate(pools):
            current = current_matrix[i]  # Get the current for this pool
            # Create appropriate current source based on current pattern
            if np.all(np.diff(current) == 0):  # If current is constant
                current_source = sim.DCSource(amplitude=current[0])
            else:  # If current is varying
                # Create a time-varying current source
                times = np.arange(0, len(current) * timestep, timestep)
                current_source = sim.StepCurrentSource(times=times, amplitudes=current)
            # Inject the current into all neurons in the pool
            current_source.inject_into(pool, location="soma")

        # Set up recording
        for pool in pools:
            pool.record("spikes")
            # Record from first two neurons for detailed analysis
            pool[0:2].record("v", locations=("dendrite", "soma"))
            pool[0:2].record(("na.m", "na.h"), locations="soma")
            pool[0:2].record(("kf.n"), locations="soma")
            pool[0:2].record(("ks.p"), locations="soma")

        # Run simulation
        sim.run(len(current_matrix[0]) * timestep)

        # Improved visualization
        n_pools = len(pools)
        time_points = np.arange(0, len(current_matrix[0]) * timestep, timestep)
        fig, axes = plt.subplots(n_pools, 2, figsize=(14, 4 * n_pools))
        if n_pools == 1:
            axes = np.array([axes])
        for i, pool in enumerate(pools):
            data = pool.get_data().segments[0]
            vm = data.filter(name="soma.v")[0]
            spikes = data.spiketrains
            # Plot membrane potential for the first neuron only
            ax_vm = axes[i, 0]
            ax_vm.plot(vm.times, vm[:, 0])
            ax_vm.set_title(f"Pool {i + 1} Membrane Potential (Neuron 0)")
            ax_vm.set_xlabel("Time (ms)")
            ax_vm.set_ylabel("mV")
            # Raster plot for all spikes
            ax_spk = axes[i, 1]
            for j, st in enumerate(spikes):
                ax_spk.vlines(st, j + 0.5, j + 1.5)
            ax_spk.set_title(f"Pool {i + 1} Spikes (Raster)")
            ax_spk.set_xlabel("Time (ms)")
            ax_spk.set_ylabel("Neuron")
        plt.tight_layout()
        plt.savefig("multi_pool_results_readable.png")
        plt.show()
        return pools

    finally:
        # Clean up simulation
        sim.end()


def plot_input_currents(current_matrix, timestep):
    n_pools = len(current_matrix)
    t_points = len(current_matrix[0])
    t = np.arange(0, t_points * timestep, timestep)
    fig, axes = plt.subplots(n_pools, 1, figsize=(10, 2 * n_pools))
    if n_pools == 1:
        axes = [axes]
    for i, current in enumerate(current_matrix):
        axes[i].plot(t, current)
        axes[i].set_title(f"Pool {i + 1} Input Current")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Current (nA)")
    plt.tight_layout()
    plt.savefig("input_currents.png")
    plt.show()


if __name__ == "__main__":
    # Get number of neurons per pool from command line argument
    neurons_per_pool = 1000

    # Example usage
    n_pools = 5  # number of motor neuron pools
    t_points = 10000  # number of time points (1000 ms / 0.05 ms = 20000)
    timestep = 0.05  # ms
    t = np.arange(0, t_points * timestep, timestep)
    current_matrix = np.zeros((n_pools, t_points))
    for i in range(n_pools):
        amplitude = np.random.uniform(
            100, 200
        )  # Random amplitude between 100 and 200 nA
        frequency = np.random.uniform(1, 10)  # Random frequency between 1 and 10 Hz
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        current_matrix[i] = (
            amplitude * np.sin(2 * np.pi * frequency * t / 1000 + phase) + 100
        )  # Convert t to seconds
    # Run simulation
    pools = run_simulation(current_matrix, neurons_per_pool=neurons_per_pool)
    # Plot and save input currents
    plot_input_currents(current_matrix, timestep)
