import sys

sys.path.append("./../")
import pyNN.neuron as sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pyNN.neuron.morphology import uniform, centre
from pyNN.parameters import IonicSpecies
import src.Classes as Classes
from src import functions


# set backend to QtAgg
plt.switch_backend("QtAgg")


def create_motor_neuron_pool(n_neurons):
    """Create a pool of motor neurons with specified parameters"""
    rng = np.random.default_rng()
    somas = functions.create_somas(n_neurons)
    dends = functions.create_dends(n_neurons, somas)
    vt = -70 + 12.35*np.exp(np.arange(n_neurons)/(n_neurons-1)*np.log(20.9/12.35))*(1+0.05*rng.normal(size=n_neurons))
    cell_type = Classes.cell_class(
        morphology=functions.soma_dend(somas, dends),
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
            "conductance_density": uniform("soma", 30),
            "vt": list(vt),
        },
        kf={
            "conductance_density": uniform("soma", 1),
            "vt": list(vt),
        },
        ks={
            "conductance_density": uniform("soma", 1),
            "vt": list(vt),
        },
        syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
    )
    return cell_type


def run_simulation(
    current_matrix,
    neurons_per_pool=100,
    timestep__ms=0.05,
    noise_mean__nA=30,
    noise_stdev__nA=30,
):
    """
    Run simulation with custom current inputs

    Parameters:
    -----------
    current_matrix : numpy.ndarray
        Matrix of shape (n_pools, t_points) containing current values
        Each row represents the current for one pool
    neurons_per_pool : int
        Number of neurons in each pool
    timestep__ms : float
        Simulation timestep in ms
    noise_mean__nA : float
        Mean of the noise current in nA
    noise_stdev__nA : float
        Standard deviation of the noise current in nA
    """

    functions.update_mod_files()
    # Validate input
    if not isinstance(current_matrix, np.ndarray):
        raise TypeError("current_matrix must be a numpy array")
    if current_matrix.ndim != 2:
        raise ValueError("current_matrix must be 2-dimensional")

    # Setup simulation with proper parameters
    sim.setup(timestep=timestep__ms)

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
            print(current)
            # Create appropriate current source based on current pattern
            if np.all(np.diff(current) == 0):  # If current is constant
                current_source = sim.DCSource(amplitude=current[0])
            else:  # If current is varying
                # Create a time-varying current source
                times = np.arange(0, len(current) * timestep__ms, timestep__ms)
                current_source = sim.StepCurrentSource(times=times, amplitudes=current)
            # Inject the current into all neurons in the pool
            current_source.inject_into(pool, location="soma")

            # add noise to each individual neuron
            for j in range(len(pool)):
                # Add Gaussian noise current to each neuron
                noise_source = sim.NoisyCurrentSource(
                    mean=noise_mean__nA,
                    stdev=noise_stdev__nA,
                    start=0.0,
                    stop=len(current) * timestep__ms,
                    dt=timestep__ms,
                )
                noise_source.inject_into([pool[j]], location="soma")

        # Set up recording
        for pool in pools:
            pool.record("spikes")
            # Record from first two neurons for detailed analysis
            pool[0:2].record("v", locations=("dendrite", "soma"))
            pool[0:2].record(("na.m", "na.h"), locations="soma")
            pool[0:2].record(("kf.n"), locations="soma")
            pool[0:2].record(("ks.p"), locations="soma")

        # Run simulation
        sim.run(len(current_matrix[0]) * timestep__ms)

        # Improved visualization
        n_pools = len(pools)
        time_points = np.arange(0, len(current_matrix[0]) * timestep__ms, timestep__ms)
        fig, axes = plt.subplots(n_pools, 2, figsize=(14, 4 * n_pools), sharex="row")
        if n_pools == 1:
            axes = np.array([axes])
        for i, pool in enumerate(pools):
            data = pool.get_data().segments[0]
            vm = data.filter(name="soma.v")[0]
            m = data.filter(name="soma.na.m")[0]
            h = data.filter(name="soma.na.h")[0]
            n = data.filter(name="soma.kf.n")[0]
            p = data.filter(name="soma.ks.p")[0]
            spikes = data.spiketrains
            # Plot membrane potential for the first neuron only
            ax_vm = axes[i, 0]
            ax_vm.plot(vm.times, vm[:, 0])
            ax_vm.set_title(f"Pool {i + 1} Membrane Potential (Neuron 0)")
            ax_vm.set_xlabel("Time (ms)")
            ax_vm.set_ylabel("mV")
            # Raster plot for all spikes
            ax_spk = axes[i, 1]

            cmap = plt.cm.turbo
            num_neurons = len(spikes)

            # Calculate the number of spikes for each neuron
            spike_counts = [len(st) for st in spikes]

            # Get min and max spike counts for this specific pool
            if spike_counts:
                pool_min = min(spike_counts)
                pool_max = max(spike_counts)
            else:
                pool_min = 0
                pool_max = 1  # Avoid division by zero if no spikes

            # Create a normalization based on spike counts
            norm = Normalize(vmin=pool_min, vmax=pool_max)

            for j, st in enumerate(spikes):
                # Assign color based on number of spikes
                spike_count = len(st)
                color = cmap(
                    norm(spike_count)
                )  # Apply colormap to normalized spike count
                ax_spk.scatter(st, np.ones(len(st)) * j, s=1, color=color)

            # set limits
            ax_spk.set_xlim(0, len(current_matrix[0]) * timestep__ms)
            ax_spk.set_title(f"Pool {i + 1} Spikes (Raster)")
            ax_spk.set_xlabel("Time (ms)")
            ax_spk.set_ylabel("Neuron")

            # Add a colorbar to show the mapping between colors and spike counts
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # cbar = plt.colorbar(sm, ax=ax_spk)
            # cbar.set_label(f'Pool {i + 1} Number of Spikes ({pool_min}-{pool_max})')
        plt.tight_layout()
        plt.savefig("multi_pool_results_readable.png")
        return pools

    finally:
        # Clean up simulation
        sim.end()


def plot_input_currents(current_matrix, timestep):
    n_pools = len(current_matrix)
    t_points = len(current_matrix[0])
    t = np.arange(0, t_points * timestep, timestep)
    fig, axes = plt.subplots(n_pools, 1, figsize=(10, 2 * n_pools), sharex=True)
    if n_pools == 1:
        axes = [axes]
    for i, current in enumerate(current_matrix):
        axes[i].plot(t, current, "o")
        axes[i].set_title(f"Pool {i + 1} Input Current")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Current (nA)")
    plt.tight_layout()
    plt.savefig("input_currents.png")


if __name__ == "__main__":
    # Get number of neurons per pool from command line argument
    neurons_per_pool = 100
    n_pools = 3

    # Simulation parameters
    timestep = 0.05  # ms
    simulation_time = 1000  # Total simulation time in ms

    noise_mean = 50  # Mean noise current (nA)
    noise_stdev = 50  # Standard deviation of noise (nA)

    # Example usage
    t_points = int(simulation_time / timestep)  # number of time points
    t = np.arange(0, simulation_time, timestep)

    current_matrix = np.zeros((n_pools, t_points))
    for i in range(n_pools):
        amplitude = np.random.uniform(100, 200)
        frequency = np.random.uniform(1, 10)  # Random frequency between 1 and 10 Hz
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase

        current_matrix[i] = (
            amplitude * np.sin(2 * np.pi * frequency * t / 1000 + phase) + 100
        )  # Convert t to seconds

    # Run simulation
    pools = run_simulation(
        current_matrix,
        neurons_per_pool=neurons_per_pool,
        timestep__ms=timestep,
        noise_mean__nA=noise_mean,
        noise_stdev__nA=noise_stdev,
    )

    # Plot and save input currents
    plot_input_currents(current_matrix, timestep)
    plt.show()
