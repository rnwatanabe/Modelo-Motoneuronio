import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import os
    import shutil
    import sys
    import platform

    sys.path.append("./../")

    import matplotlib.pyplot as plt
    import numpy as np
    import pyNN
    import pyNN.neuron as sim
    import pyNN.space as space
    import pandas as pd

    from neuroml import Morphology, Segment, Point3DWithDiam as P
    from neuron import h, hclass
    from pyNN import neuron
    from pyNN.models import BaseCellType
    from pyNN.morphology import (
        NeuroMLMorphology,
        NeuriteDistribution,
        Morphology as Morph,
        IonChannelDistribution,
    )
    from pyNN.neuron import NativeCellType
    from pyNN.neuron.cells import RandomSpikeSource, _new_property
    from pyNN.neuron.morphology import (
        uniform,
        random_section,
        random_placement,
        at_distances,
        apical_dendrites,
        dendrites,
        centre,
    )
    from pyNN.neuron.simulator import state
    from pyNN.parameters import IonicSpecies
    from pyNN.random import RandomDistribution, NumpyRNG
    from pyNN.space import Grid2D, RandomStructure, Sphere
    from pyNN.standardmodels import (
        StandardIonChannelModel,
        build_translations,
        StandardCellType,
        StandardModelType,
    )
    from pyNN.standardmodels.cells import SpikeSourceGamma, MultiCompartmentNeuron as mc
    from pyNN.utility.build import compile_nmodl
    from pyNN.utility.plotting import Figure, Panel
    import src.Classes as Classes
    import src.functions as funçoes
    from src.functions import neuromuscular_system, soma_força

    return (
        Classes,
        Figure,
        IonicSpecies,
        Panel,
        centre,
        compile_nmodl,
        funçoes,
        h,
        neuromuscular_system,
        np,
        os,
        pd,
        platform,
        plt,
        shutil,
        sim,
        soma_força,
        uniform,
    )


@app.cell
def _(compile_nmodl, h, os, platform, shutil):
    files = os.listdir("../src/")
    if platform.system() == "Linux":
        for _filename in files:
            if _filename.endswith(".mod"):
                shutil.copyfile(
                    f"./../src/{_filename}",
                    f"./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/{_filename}",
                )
        compile_nmodl("./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl")
        h.nrn_load_dll("./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/mn.o")
    if platform.system() == "Windows":
        for _filename in files:
            if _filename.endswith(".mod"):
                shutil.copyfile(
                    f"./../src/{_filename}",
                    f"../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/{_filename}",
                )
        compile_nmodl("../modelpynn/Lib/site-packages/pyNN/neuron/nmodl")
        h.nrn_load_dll("../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.o")
    return


@app.cell
def _(
    Classes,
    IonicSpecies,
    centre,
    funçoes,
    h,
    neuromuscular_system,
    np,
    sim,
    uniform,
):
    timestep = 0.05
    sim.setup(timestep=timestep)
    Tf = 30000
    _n = 100
    somas = funçoes.create_somas(_n)
    dends = funçoes.create_dends(_n, somas)
    cell_type = Classes.cell_class(
        morphology=funçoes.soma_dend(somas, dends),
        cm=1,
        Ra=0.07,
        ionic_species={
            "na": IonicSpecies("na", reversal_potential=50),
            "ks": IonicSpecies("ks", reversal_potential=-80),
            "kf": IonicSpecies("kf", reversal_potential=-80),
        },
        pas_soma={"conductance_density": uniform("soma", 0.0007), "e_rev": -70},
        pas_dend={"conductance_density": uniform("dendrite", 0.0007), "e_rev": -70},
        na={
            "conductance_density": uniform("soma", 10),
            "vt": list(np.linspace(-57.65, -53, _n)),
        },
        kf={
            "conductance_density": uniform("soma", 1),
            "vt": list(np.linspace(-57.65, -53, _n)),
        },
        ks={
            "conductance_density": uniform("soma", 0.5),
            "vt": list(np.linspace(-57.65, -53, _n)),
        },
        syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
    )
    cells = sim.Population(_n, cell_type, initial_values={"v": list(-70 * np.ones(_n))})
    muscle_units, force_objects, neuromuscular_junctions = neuromuscular_system(
        cells, _n, h, Umax=1600
    )
    np.random.seed(237371)
    spike_source = sim.Population(400, Classes.SpikeSourceGammaStart(alpha=1))
    syn = sim.StaticSynapse(weight=0.6, delay=0.2)
    input_conns = sim.Projection(
        spike_source,
        cells,
        sim.FixedProbabilityConnector(0.3, location_selector="dendrite"),
        syn,
        receptor_type="syn",
    )
    spike_source.record("spikes")
    cells.record("spikes")
    cells[0:2].record("v", locations=("dendrite", "soma"))
    cells[0:2].record(("na.m", "na.h"), locations="soma")
    cells[0:2].record("kf.n", locations="soma")
    cells[0:2].record("ks.p", locations="soma")
    f = dict()
    for _i in range(_n):
        f[_i] = h.Vector().record(force_objects[_i]._ref_F)
    return Tf, cells, f, force_objects, spike_source, timestep


@app.cell
def _(
    Classes,
    Figure,
    Panel,
    Tf,
    cells,
    f,
    force_objects,
    h,
    np,
    os,
    pd,
    plt,
    sim,
    soma_força,
    spike_source,
    timestep,
):
    refs = [55, 110, 220, 330, 440, 550, 660, 770, 880, 990, 1100]
    _n = 0
    for ref in refs:
        sim.run(
            Tf,
            callbacks=[
                Classes.SetRateIntControl(spike_source, cells, force_objects, ref=ref)
            ],
        )
        print("fim simulação")
        força_total = soma_força(force_objects, h, f)
        rate = 10 + ((ref - força_total) * 0.01 + 0.1 * np.cumsum(força_total) * 0.05)
        t = np.arange(0, Tf, timestep)
        plt.plot(t, força_total)
        plt.show()
        new_folder = "força2.Umax=1600"
        os.makedirs(new_folder, exist_ok=True)
        df = pd.DataFrame({"time": t, "força_total": força_total, "rate": rate})
        _filename = os.path.join(new_folder, f"Ca_forca_ref{ref}_Umax=1600.csv")
        df.to_csv(_filename, index=False)
        data_source = spike_source.get_data().segments[_n]
        data = cells.get_data().segments[_n]
        spike_df = pd.DataFrame(
            [
                {"neuron_id": neuron_id, "spike_time": spike_time}
                for neuron_id, spikes in enumerate(data_source.spiketrains)
                for spike_time in spikes
            ]
        )
        new_folder1 = "spikedatasource2.Umax=1600"
        os.makedirs(new_folder1, exist_ok=True)
        _filename = os.path.join(new_folder1, f"Ca_spike_data_ref_{ref}_Umax=1600.csv")
        spike_df.to_csv(_filename, index=False)
        plt.scatter(
            spike_df["spike_time"], spike_df["neuron_id"], s=4, label=f"ref={ref}"
        )
        plt.show()
        cell_spike_df = pd.DataFrame(
            [
                {"neuron_id": neuron_id, "spike_time": spike_time}
                for neuron_id, spikes in enumerate(data.spiketrains)
                for spike_time in spikes
            ]
        )
        new_folder2 = "spikedata2.Umax=1600"
        os.makedirs(new_folder2, exist_ok=True)
        _filename = os.path.join(new_folder2, f"Ca_cell_spike_ref_{ref}_Umax=1600.csv")
        cell_spike_df.to_csv(_filename, index=False)
        plt.scatter(
            cell_spike_df["spike_time"],
            cell_spike_df["neuron_id"],
            s=4,
            label=f"ref={ref}",
        )
        plt.show()
        data = cells.get_data().segments[0]
        vm = data.filter(name="soma.v")[0]
        Figure(Panel(vm, ylabel="Membrane potential (mV)", xticks=True, yticks=True))
        plt.show()
        sim.reset()
        for _i in range(100):
            force_objects[_i].F = 0
            force_objects[_i].x1 = 0
            force_objects[_i].x2 = 0
        _n = _n + 1
    return


if __name__ == "__main__":
    app.run()
