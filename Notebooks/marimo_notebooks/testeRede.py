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
    from pyNN.standardmodels import StandardIonChannelModel, build_translations
    from pyNN.neuron import NativeCellType
    import shutil
    import os
    from neuron import h, hclass
    from pyNN.utility.build import compile_nmodl
    from pyNN.standardmodels.cells import MultiCompartmentNeuron as mc
    import platform
    import src.Classes as Classes
    import src.functions as funçoes

    return (
        Classes,
        Figure,
        IonicSpecies,
        Panel,
        centre,
        funçoes,
        np,
        sim,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Configuração para os sistemas, compilação e carregamento de um arquivo de modelo (mn.mod) em um ambiente de simulação de redes neurais utilizando o pacote PyNN com o mecanismo de simulação NEURON
        """
    )
    return


@app.cell
def _(Classes, IonicSpecies, centre, funçoes, np, uniform):
    n = 100

    somas = funçoes.create_somas(n)
    dends = funçoes.create_dends(n, somas)

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
    return cell_type, n


@app.cell
def _(cell_type, n, np, sim):
    cells = sim.Population(n, cell_type, initial_values={"v": list(-70 * np.ones(100))})
    return (cells,)


@app.cell
def _(cells):
    cells[0]._cell.synaptic_receptors
    return


@app.cell
def _(cells, funçoes, np, sim):
    np.random.seed(26278342)
    spike_source = sim.Population(
        400, sim.SpikeSourceArray(spike_times=funçoes.generate_spike_times)
    )
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
def _(spike_source):
    spike_source
    return


@app.cell
def _(sim):
    step_current_dend = sim.DCSource(amplitude=7000, start=0, stop=50)
    # step_current.inject_into(cells[1:2], location=apical_dendrites(fraction_along=0.9))
    # step_current.inject_into(cells[1:2], location=random(after_branch_point(3)(apical_dendrites))
    # step_current_dend.inject_into(cells[0:2], location='dendrite')
    return


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
def _(sim):
    sim.run(100)
    return


@app.cell
def _(Figure, Panel, cells):
    _figure_filename = "teste.png"
    _data = cells.get_data().segments[0]
    _vm = _data.filter(name="soma.v")[0]
    _m = _data.filter(name="soma.na.m")[0]
    _h = _data.filter(name="soma.na.h")[0]
    n_1 = _data.filter(name="soma.kf.n")[0]
    _p = _data.filter(name="soma.ks.p")[0]
    Figure(
        Panel(_vm, ylabel="Membrane potential (mV)", xticks=True, yticks=True),
        Panel(_m, ylabel="m state", xticks=True, yticks=True),
        Panel(_h, ylabel="h state", xticks=True, yticks=True),
        Panel(n_1, ylabel="n state", xticks=True, yticks=True),
        Panel(_p, ylabel="p state", xticks=True, yticks=True),
        Panel(_data.spiketrains, xlabel="Time (ms)", xticks=True, yticks=True),
    ).save(_figure_filename)
    return


@app.cell
def _(Figure, Panel, cells, sim):
    sim.run(100)
    _figure_filename = "teste.png"
    _data = cells.get_data().segments[0]
    _vm = _data.filter(name="soma.v")[0]
    _m = _data.filter(name="soma.na.m")[0]
    _h = _data.filter(name="soma.na.h")[0]
    n_2 = _data.filter(name="soma.kf.n")[0]
    _p = _data.filter(name="soma.ks.p")[0]
    Figure(
        Panel(_vm, ylabel="Membrane potential (mV)", xticks=True, yticks=True),
        Panel(_m, ylabel="m state", xticks=True, yticks=True),
        Panel(_h, ylabel="h state", xticks=True, yticks=True),
        Panel(n_2, ylabel="n state", xticks=True, yticks=True),
        Panel(_p, ylabel="p state", xticks=True, yticks=True),
        Panel(_data.spiketrains, xlabel="Time (ms)", xticks=True, yticks=True),
    ).save(_figure_filename)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
