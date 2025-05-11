import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import sys

    sys.path.append("./../")

    from pyNN.random import RandomDistribution, NumpyRNG
    from pyNN import neuron
    import pyNN.space as space
    import pyNN
    import pyNN.neuron as sim
    import numpy as np
    from pyNN.utility.plotting import Figure, Panel
    from pyNN.space import Grid2D, RandomStructure, Sphere
    import matplotlib.pyplot as plt
    from neuroml import Morphology, Segment, Point3DWithDiam as P
    from pyNN.morphology import (
        NeuroMLMorphology,
        NeuriteDistribution,
        Morphology as Morph,
        IonChannelDistribution,
    )
    from pyNN.neuron.morphology import (
        uniform,
        random_section,
        random_placement,
        at_distances,
        apical_dendrites,
        dendrites,
        centre,
    )
    from pyNN.parameters import IonicSpecies
    from pyNN.standardmodels import (
        StandardIonChannelModel,
        build_translations,
        StandardCellType,
    )
    from pyNN.standardmodels.cells import SpikeSourceGamma
    from pyNN.neuron import NativeCellType
    import shutil
    import os
    from neuron import h, hclass

    h.nrn_load_dll("../src/gammapointprocess.o")

    from pyNN.utility.build import compile_nmodl
    from pyNN.standardmodels.cells import MultiCompartmentNeuron as mc
    import platform
    from pyNN.neuron.cells import RandomSpikeSource
    import src.Classes as Classes
    import src.funçoes as funçoes

    return (
        Classes,
        Figure,
        IonicSpecies,
        Panel,
        centre,
        compile_nmodl,
        funçoes,
        h,
        np,
        platform,
        shutil,
        sim,
        uniform,
    )


@app.cell
def _(compile_nmodl, h, platform, shutil):
    if platform.system() == "Linux":
        shutil.copyfile(
            "./../src/mn.mod",
            "./../modelpynn/lib/python3.10/site-packages/pyNN/neuron/nmodl/mn.mod",
        )
        shutil.copyfile(
            "./../src/gammapointprocess.mod",
            "./../modelpynn/lib/python3.10/site-packages/pyNN/neuron/nmodl/gammapointprocess.mod",
        )
        compile_nmodl("./../modelpynn/lib/python3.10/site-packages/pyNN/neuron/nmodl/")
        h.nrn_load_dll(
            "./../modelpynn/lib/python3.10/site-packages/pyNN/neuron/nmodl/mn.o"
        )
    if platform.system() == "Windows":
        shutil.copyfile(
            "../src/mn.mod", "../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.mod"
        )
        shutil.copyfile(
            "../src/gammapointprocess.mod",
            "../modelpynn/Lib/site-packages/pyNN/neuron/nmodl/gammapointprocess.mod",
        )
        compile_nmodl("../modelpynn/Lib/site-packages/pyNN/neuron/nmodl")
        h.nrn_load_dll("modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.o")
    return


@app.cell
def _(funçoes, sim):
    sim.setup(timestep=0.05)

    n = 100
    somas = funçoes.create_somas(n)
    dends = funçoes.create_dends(n, somas)
    return dends, n, somas


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
def _(Classes, cells, np, sim):
    np.random.seed(26278342)
    spike_source = sim.Population(400, Classes.SpikeSourceGammaStart(alpha=1))
    # start=RandomDistribution('uniform', [0, 3.0], rng=NumpyRNG(seed=4242))))
    syn = sim.StaticSynapse(weight=0.6, delay=0.2)
    input_conns = sim.Projection(
        spike_source,
        cells,
        sim.FixedProbabilityConnector(0.3, location_selector="dendrite"),
        syn,
        receptor_type="syn",
    )
    return (spike_source,)


@app.cell
def _(cells, spike_source):
    spike_source.record("spikes")
    cells.record("spikes")
    cells[0:2].record("v", locations=("dendrite", "soma"))
    cells[0:2].record(("na.m", "na.h"), locations="soma")
    cells[0:2].record(("kf.n"), locations="soma")
    cells[0:2].record(("ks.p"), locations="soma")
    return


@app.cell
def _(Classes, cells, sim, spike_source):
    sim.run(1000, callbacks=[Classes.SetRate(spike_source, cells)])
    return


@app.cell
def _(Figure, Panel, cells, spike_source):
    figure_filename = "teste.png"
    data_source = spike_source.get_data().segments[0]
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
    return (data,)


@app.cell
def _(data):
    data.spiketrains[10].as_array()
    return


if __name__ == "__main__":
    app.run()
