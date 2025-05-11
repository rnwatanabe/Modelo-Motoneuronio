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

    from pyNN.utility.build import compile_nmodl

    if platform.system() == "Linux":
        shutil.copyfile(
            "./../src/mn.mod",
            "./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/mn.mod",
        )
        shutil.copyfile(
            "./../src/muscle_unit.mod",
            "./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/muscle_unit.mod",
        )
        shutil.copyfile(
            "./../src/muscle_unit_calcium.mod",
            "./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/muscle_unit_calcium.mod",
        )
        shutil.copyfile(
            "./../src/gammapointprocess.mod",
            "./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/gammapointprocess.mod",
        )
        compile_nmodl("./../.venv/lib/python3.10/site-packages/pyNN/neuron/nmodl/")
        # h.nrn_load_dll('./../modelpynn/lib/python3.10/site-packages/pyNN/neuron/nmodl/mn.o')
    if platform.system() == "Windows":
        shutil.copyfile(
            "../src/mn.mod", "../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.mod"
        )
        shutil.copyfile(
            "./../src/muscle_unit.mod",
            "./../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/muscle_unit.mod",
        )
        shutil.copyfile(
            "./../src/muscle_unit_calcium.mod",
            "./../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/muscle_unit_calcium.mod",
        )
        shutil.copyfile(
            "../src/gammapointprocess.mod",
            "../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/gammapointprocess.mod",
        )
        compile_nmodl("../modelpynn/Lib/site-packages/pyNN/neuron/nmodl")
        h.nrn_load_dll("modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.o")

    import matplotlib.pyplot as plt
    import numpy as np
    import pyNN
    import pyNN.neuron as sim
    import pyNN.space as space

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

    from pyNN.utility.plotting import Figure, Panel
    import src.Classes as Classes
    import src.functions as funçoes
    from src.functions import neuromuscular_system, soma_força

    # %matplotlib widget
    return (
        Classes,
        Figure,
        IonicSpecies,
        Panel,
        centre,
        funçoes,
        h,
        neuromuscular_system,
        np,
        plt,
        sim,
        uniform,
    )


@app.cell
def _(funçoes, sim):
    timestep = 0.05
    sim.setup(timestep=timestep)
    Tf = 1000
    n = 100
    somas = funçoes.create_somas(n)
    dends = funçoes.create_dends(n, somas)
    return Tf, dends, n, somas, timestep


@app.cell
def _(Classes, IonicSpecies, centre, dends, funçoes, np, somas, uniform):
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
            "vt": list(np.linspace(-57.65, -53, 100)),
        },
        kf={
            "conductance_density": uniform("soma", 1),
            "vt": list(np.linspace(-57.65, -53, 100)),
        },
        ks={
            "conductance_density": uniform("soma", 0.5),
            "vt": list(np.linspace(-57.65, -53, 100)),
        },
        syn={"locations": centre("dendrite"), "e_syn": 0, "tau_syn": 0.6},
    )
    return (cell_type,)


@app.cell
def _(cell_type, n, np, sim):
    cells = sim.Population(n, cell_type, initial_values={"v": list(-70 * np.ones(n))})
    return (cells,)


@app.cell
def _(cells, h, n, neuromuscular_system):
    muscle_units, force_objects, neuromuscular_junctions = neuromuscular_system(
        cells, n, h
    )
    return (force_objects,)


@app.cell
def _(Classes, cells, np, sim):
    np.random.seed(26278342)
    spike_source = sim.Population(400, Classes.SpikeSourceGammaStart(alpha=1))
    # start=RandomDistribution('uniform', [0, 3.0], rng=NumpyRNG(seed=4242))))
    syn = sim.StaticSynapse(weight=0.6, delay=0.2)
    # nmj = sim.StaticSynapse(weight=1, delay=0.2)
    input_conns = sim.Projection(
        spike_source,
        cells,
        sim.FixedProbabilityConnector(0.3, location_selector="dendrite"),
        syn,
        receptor_type="syn",
    )
    return (spike_source,)


@app.cell
def _(cells, force_objects, h, n, spike_source):
    spike_source.record("spikes")
    cells.record("spikes")
    cells[0:2].record("v", locations=("dendrite", "soma"))
    cells[0:2].record(("na.m", "na.h"), locations="soma")
    cells[0:2].record(("kf.n"), locations="soma")
    cells[0:2].record(("ks.p"), locations="soma")
    f = dict()
    cat = dict()
    for i in range(n):
        f[i] = h.Vector().record(force_objects[i]._ref_F)
        cat[i] = h.Vector().record(force_objects[i]._ref_CaT)
    return cat, f


@app.cell
def _():
    # soma_força(force_objects, 100)
    # f, força_total = soma_força(force_objects,100)
    return


@app.cell
def _(Classes, Tf, cells, sim, spike_source):
    sim.run(Tf, callbacks=[Classes.SetRate(spike_source, cells)])
    return


@app.cell
def _(Figure, Panel, Tf, cat, cells, f, np, plt, spike_source, timestep):
    figure_filename = "teste.png"
    data_source = spike_source.get_data().segments
    data = cells.get_data().segments[0]
    vm = data.filter(name="soma.v")[0]
    m = data.filter(name="soma.na.m")[0]
    h_1 = data.filter(name="soma.na.h")[0]
    n_1 = data.filter(name="soma.kf.n")[0]
    p = data.filter(name="soma.ks.p")[0]
    Figure(
        Panel(data_source.spiketrains, xlabel="Time (ms)", xticks=True, yticks=True),
        Panel(vm, ylabel="Membrane potential (mV)", xticks=True, yticks=True),
        Panel(m, ylabel="m state", xticks=True, yticks=True),
        Panel(h_1, ylabel="h state", xticks=True, yticks=True),
        Panel(n_1, ylabel="n state", xticks=True, yticks=True),
        Panel(p, ylabel="p state", xticks=True, yticks=True),
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True, yticks=True),
    ).save(figure_filename)
    plt.figure()
    plt.plot(np.arange(0, Tf + timestep, timestep), f[0])
    plt.show()
    plt.figure()
    plt.plot(np.arange(0, Tf + timestep, timestep), cat[0])
    plt.show()
    return (data,)


@app.cell
def _(data):
    data.spiketrains[0].as_array()
    return


if __name__ == "__main__":
    app.run()
