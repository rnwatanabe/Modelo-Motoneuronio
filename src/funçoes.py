from neuroml import Morphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology, NeuriteDistribution, Morphology as Morph, IonChannelDistribution
import pyNN.neuron as sim
import numpy as np

def create_somas(n):
    somas = []
    for i in range(n):
        diameter = 77.5 + i * ((82.5-77.5)/n)
        y = 18- i * (18/n)
        soma= Segment(proximal=P(x=diameter, y=y, z=0, diameter=diameter),
                       distal=P(x=0, y=0, z=0, diameter=diameter),
                       name="soma", id=0)
        somas.append(soma)
    return somas

def create_dends(n,somas):
     dends=[]
     for i in range(n):
        y = 18- i * (18/n)
        diameter = 41.5 + i * ((62.5-41.5)/n)
        x_distal= -5500 + i * ((-6789+5500)/n)
        dend = Segment(proximal=P(x=0, y=y, z=0, diameter=diameter),
               distal=P(x=x_distal, y=y, z=0, diameter=diameter),
               name="dendrite",
               parent=somas[i], id=1)
        dends.append(dend)
     return dends

def soma_dend(somas, dends):
    combined= []
    first_soma = somas[0]
    for i in range(len(dends)):
        combineds=NeuroMLMorphology(Morphology(segments=(somas[i], 
                                                dends[i])))
        combined.append(combineds)
    return combined

def generate_spike_times(i):
    input_rate = 83
    Tf = 100
    number = int(Tf * input_rate / 1000.0)
    gen = lambda: sim.Sequence(np.add.accumulate(np.random.exponential(1000.0 / input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()

# somas= create_somas(n)
# dendrites = create_dends(n,somas)
# neurons= soma_dend(somas, dendrites) 