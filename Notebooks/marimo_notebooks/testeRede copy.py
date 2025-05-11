import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
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

    import os
    from pyNN.standardmodels.cells import MultiCompartmentNeuron as mc

    return (
        Figure,
        IonChannelDistribution,
        IonicSpecies,
        Morphology,
        NeuroMLMorphology,
        P,
        Panel,
        Segment,
        StandardIonChannelModel,
        build_translations,
        centre,
        np,
        sim,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Configuration for systems, compilation and loading of a model file (mn.mod) in a neural network simulation environment using the PyNN package with the NEURON simulation engine

        """
    )
    return


@app.cell
def _():
    import platform
    import shutil
    from pyNN.utility.build import compile_nmodl
    from neuron import h, hclass
    from pathlib import Path

    if platform.system() == "Linux":
        compile_nmodl("src")
        _h.nrn_load_dll("src/mn.o")
    if platform.system() == "Windows":
        shutil.copyfile(
            "mn.mod", "modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.mod"
        )
        compile_nmodl("modelpynn/Lib/site-packages/pyNN/neuron/nmodl")
        _h.nrn_load_dll("modelpynn/Lib/site-packages/pyNN/neuron/nmodl/mn.o")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instalation of the potassium channels
        """
    )
    return


@app.cell
def _(IonChannelDistribution, StandardIonChannelModel, build_translations):
    class KsChannel(StandardIonChannelModel):
        default_parameters = {
            "conductance_density": 0.12,  # uniform('all', 0.12),
            "e_rev": -80,
            "vt": -57.65,
        }

        translations = build_translations(
            ("conductance_density", "gk_slow"),
            ("e_rev", "eks"),
            ("vt", "vt"),
        )
        variable_translations = {
            "p": ("motoneuron", "p"),
        }
        default_initial_values = {
            "p": 1,  # initial value for gating variable m
        }
        units = {
            "iks": "mA/cm2",
            "p": "dimensionless",
        }
        recordable = ["iks", "p"]
        model = "motoneuron"
        conductance_density_parameter = "gk_slow"

        def get_schema(self):
            return {
                "conductance_density": IonChannelDistribution,
                "e_rev": float,
                "vt": float,
            }

    return (KsChannel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instalation of the fast potassium channels
        """
    )
    return


@app.cell
def _(IonChannelDistribution, StandardIonChannelModel, build_translations):
    class KfChannel(StandardIonChannelModel):
        default_parameters = {
            "conductance_density": 0.12,  # uniform('all', 0.12),
            "e_rev": -80,
            "vt": -57.65,
        }

        recordable = ["ikf", "n"]
        translations = build_translations(
            ("conductance_density", "gk_fast"),
            ("e_rev", "ekf"),
            ("vt", "vt"),
        )
        variable_translations = {
            "n": ("motoneuron", "n"),
        }
        default_initial_values = {
            "n": 1,  # initial value for gating variable m
        }
        units = {
            "ikf": "mA/cm2",
            "n": "dimensionless",
        }
        model = "motoneuron"
        conductance_density_parameter = "gk_fast"

        def get_schema(self):
            return {
                "conductance_density": IonChannelDistribution,
                "e_rev": float,
                "vt": float,
            }

    return (KfChannel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instalation of the sodium channels
        """
    )
    return


@app.cell
def _(IonChannelDistribution, StandardIonChannelModel, build_translations):
    class NaChannel(StandardIonChannelModel):
        default_parameters = {
            "conductance_density": 0.12,  # uniform('all', 0.12),
            "e_rev": 50,
            "vt": -57.65,
        }

        default_initial_values = {
            "m": 1.0,  # initial value for gating variable m
            "h": 0.0,  # initial value for gating variable h
        }
        recordable = ["ina", "m", "h"]
        units = {
            "ina": "mA/cm2",
            "m": "dimensionless",
            "h": "dimensionless",
        }
        translations = build_translations(
            ("conductance_density", "gna"),
            ("e_rev", "ena"),
            ("vt", "vt"),
        )
        variable_translations = {
            "h": ("motoneuron", "h"),
            "m": ("motoneuron", "m"),
            "ina": ("motoneuron", "ina"),
        }
        model = "motoneuron"
        conductance_density_parameter = "gna"

        def get_schema(self):
            return {
                "conductance_density": IonChannelDistribution,
                "e_rev": float,
                "vt": float,
            }

    return (NaChannel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The PassiveLeak class models a passive leak ion channel, which is a type of channel that allows constant passage of ions through the cell membrane, regardless of the cell state (i.e., it does not depend on voltage or other state variables for its opening or closing).
        """
    )
    return


@app.cell
def _(StandardIonChannelModel, build_translations):
    class PassiveLeak(StandardIonChannelModel):
        translations = build_translations(
            ("conductance_density", "gl"),
            ("e_rev", "el"),
        )
        variable_translations = {}
        model = "motoneuron"
        conductance_density_parameter = "gl"

    return (PassiveLeak,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Modeling a neuronal cell using segments (soma and dendrites) to define the neuron morphology.
        Using the cell_class class to configure the neuron with different ion channels and properties.
        """
    )
    return


@app.cell
def _(Morphology, NeuroMLMorphology, P, Segment):
    n = 100

    def create_somas(n):
        somas = []
        for i in range(n):
            diameter = 77.5 + i * ((82.5 - 77.5) / n)
            y = i * (18 / n)
            soma = Segment(
                proximal=P(x=diameter, y=y, z=0, diameter=diameter),
                distal=P(x=0, y=y, z=0, diameter=diameter),
                name="soma",
                id=0,
            )
            somas.append(soma)
        return somas

    def create_dends(n, somas):
        dends = []
        for i in range(n):
            y = i * (18 / n)
            diameter = 41.5 + i * ((62.5 - 41.5) / n)
            x_distal = -5500 + i * ((-6789 + 5500) / n)
            dend = Segment(
                proximal=P(x=0, y=y, z=0, diameter=diameter),
                distal=P(x=x_distal, y=y, z=0, diameter=diameter),
                name="dendrite",
                parent=somas[i],
                id=1,
            )
            dends.append(dend)
        return dends

    def soma_dend(somas, dends):
        combined = []
        for i in range(len(dends)):
            combineds = NeuroMLMorphology(Morphology(segments=(somas[i], dends[i])))
            combined.append(combineds)
        return combined

    somas = create_somas(n)
    _dendrites = create_dends(n, somas)
    neurons = soma_dend(somas, _dendrites)
    neurons
    somas[2]
    return n, neurons


@app.cell
def _(KfChannel, KsChannel, NaChannel, PassiveLeak, sim):
    class cell_class(sim.MultiCompartmentNeuron):
        def __init__(self, **parameters):
            self.label = "mn1"
            self.ion_channels = {
                "pas_soma": PassiveLeak,
                "pas_dend": sim.PassiveLeak,
                "na": NaChannel,
                "kf": KfChannel,
                "ks": KsChannel,
            }
            self.units = {
                "v": "mV",
                "gsyn_exc": "uS",
                "gsyn_inh": "uS",
                "na.m": "dimensionless",
                "na.h": "dimensionless",
                "kf.n": "dimensionless",
                "ks.p": "dimensionless",
                "na.ina": "mA/cm2",
                "kf.ikf": "mA/cm2",
                "ks.iks": "mA/cm2",
            }
            self.post_synaptic_entities = {"syn": sim.CondExpPostSynapticResponse}

            super(cell_class, self).__init__(**parameters)

    return (cell_class,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The code defines a cell_class assignment called cell_type, configuring the morphology, electrical properties, ion channels and synapses of the neuron.
        """
    )
    return


@app.cell
def _(IonicSpecies, cell_class, centre, n, neurons, np, uniform):
    cell_type = cell_class(
        morphology=neurons,
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
    print(n)
    print(cell_type)
    return (cell_type,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Creates a population of 2 neurons, with action potential ('v') of -70 mV
        """
    )
    return


@app.cell
def _(cell_type, n, np, sim):
    cells = sim.Population(n, cell_type, initial_values={"v": list(-70 * np.ones(100))})
    return


@app.cell
def _(cell_type, n, np, sim):
    cells_1 = sim.Population(
        n, cell_type, initial_values={"v": list(-70 * np.ones(100))}
    )
    return


@app.cell
def _(cell_type, n, np, sim):
    cells_2 = sim.Population(
        n, cell_type, initial_values={"v": list(-70 * np.ones(100))}
    )
    return (cells_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Acessa os receptores sinapticos da primeira celula 
        """
    )
    return


@app.cell
def _(cells_2):
    cells_2[0]._cell.synaptic_receptors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Função para gerar tempos de disparos para os neurônios
        """
    )
    return


@app.cell
def _(np, sim):
    def generate_spike_times(i):
        input_rate = 83
        Tf = 100
        number = int(Tf * input_rate / 1000.0)
        gen = lambda: sim.Sequence(
            np.add.accumulate(np.random.exponential(1000.0 / input_rate, size=number))
        )
        if hasattr(i, "__len__"):
            return [gen() for j in i]
        else:
            return gen()

    return (generate_spike_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Uma população de 400 fontes de disparo é criada, 'sim.Projection' cria projeções de conexões(dendriticas) entre populações de neurônios.A probabilidade de uma conexão ser estabelecida entre dois neurônios (30%).
        """
    )
    return


@app.cell
def _(cells_2, generate_spike_times, np, sim):
    np.random.seed(26278342)
    spike_source = sim.Population(
        400, sim.SpikeSourceArray(spike_times=generate_spike_times)
    )
    syn = sim.StaticSynapse(weight=0.6, delay=0.2)
    input_conns = sim.Projection(
        spike_source,
        cells_2,
        sim.FixedProbabilityConnector(0.3, location_selector="dendrite"),
        syn,
        receptor_type="syn",
    )
    return (spike_source,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        população de fontes de disparo 
        """
    )
    return


@app.cell
def _(spike_source):
    spike_source
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Corrente contínua nos neuronios
        """
    )
    return


@app.cell
def _(sim):
    step_current_dend = sim.DCSource(amplitude=7000, start=0, stop=50)
    # step_current.inject_into(cells[1:2], location=apical_dendrites(fraction_along=0.9))
    # step_current.inject_into(cells[1:2], location=random(after_branch_point(3)(apical_dendrites))
    # step_current_dend.inject_into(cells[0:2], location='dendrite')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        configuração da simulação para registrar vários tipos de dados durante a simulação
        """
    )
    return


@app.cell
def _(cells_2, spike_source):
    spike_source.record("spikes")
    cells_2.record("spikes")
    cells_2[0:2].record("v", locations=("dendrite", "soma"))
    cells_2[0:2].record(("na.m", "na.h"), locations="soma")
    cells_2[0:2].record("kf.n", locations="soma")
    cells_2[0:2].record("ks.p", locations="soma")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        execução da simulção, com duração de 100 ms
        """
    )
    return


@app.cell
def _(sim):
    sim.run(100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Gerar e salvar a figura com os resultados da simulação neural
        """
    )
    return


@app.cell
def _(Figure, Panel, cells_2):
    _figure_filename = "teste.png"
    _data = cells_2.get_data().segments[0]
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
def _(Figure, Panel, cells_2, sim):
    sim.run(100)
    _figure_filename = "teste.png"
    _data = cells_2.get_data().segments[0]
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
