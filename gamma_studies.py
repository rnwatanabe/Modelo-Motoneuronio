import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import os
    import shutil
    import sys
    import platform
    sys.path.append('./../')



    import matplotlib.pyplot as plt
    import numpy as np
    import pyNN
    import pyNN.neuron as sim
    import pyNN.space as space
    import pandas as pd


    from neuroml import Morphology, Segment, Point3DWithDiam as P
    from neuron import h, nrn, hclass
    from pyNN import neuron
    from pyNN.models import BaseCellType
    from pyNN.morphology import NeuroMLMorphology, NeuriteDistribution, Morphology as Morph, IonChannelDistribution
    from pyNN.neuron import NativeCellType
    from pyNN.neuron.cells import RandomSpikeSource, _new_property
    from pyNN.neuron.morphology import uniform, random_section, random_placement, at_distances, apical_dendrites, dendrites, centre
    from pyNN.neuron.simulator import state
    from pyNN.parameters import IonicSpecies
    from pyNN.random import RandomDistribution, NumpyRNG
    from pyNN.space import Grid2D, RandomStructure, Sphere
    from pyNN.standardmodels import StandardIonChannelModel, build_translations, StandardCellType, StandardModelType
    from pyNN.standardmodels.cells import SpikeSourceGamma, MultiCompartmentNeuron as mc
    from pyNN.utility.build import compile_nmodl
    from pyNN.utility.plotting import Figure, Panel
    import src.Classes as Classes
    import src.functions as functions
    from src.functions import neuromuscular_system, sum_force, neuromuscular_system

    import platform

    return Classes, pd, sim


@app.cell
def _():
    return


@app.cell
def _(Classes, pd, sim):
    spike_source1 = sim.Population(1, Classes.SpikeSourceGammaStart(alpha=1))
    spike_source2 = sim.Population(1, Classes.SpikeSourceGammaStart(alpha=1))
    spike_source1.set(alpha=1)
    spike_source2.set(alpha=1)
    # spike_source.set(beta=feedforward+1e-6)    
    timestep = 0.05
    sim.setup(timestep=timestep)
    sim.run(10000, callbacks=[Classes.SetRateGamma(spike_source1, rate=100),
                              Classes.SetRateGamma(spike_source2, rate=100)])

    data_source1 = spike_source1.get_data().segments[0]
    data_source2 = spike_source2.get_data().segments[0]



            #teste spike_datasource
    spike_df1 = pd.DataFrame([{"neuron_id": neuron_id, "spike_time": spike_time}
                            for neuron_id, spikes in enumerate(data_source1.spiketrains)
                            for spike_time in spikes])

    spike_df2 = pd.DataFrame([{"neuron_id": neuron_id, "spike_time": spike_time}
                            for neuron_id, spikes in enumerate(data_source2.spiketrains)
                            for spike_time in spikes])

    return (spike_df1,)


@app.cell
def _(spike_df1):
    spike_df1
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
