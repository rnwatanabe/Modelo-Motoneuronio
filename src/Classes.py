import numpy as np
import pyNN.neuron as sim
from neuron import h, hclass
from pyNN.morphology import IonChannelDistribution
from pyNN.neuron.simulator import state
from pyNN.standardmodels import StandardIonChannelModel, build_translations
from pyNN.standardmodels.cells import SpikeSourceGamma


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


class PassiveLeak(StandardIonChannelModel):
    translations = build_translations(
        ("conductance_density", "gl"),
        ("e_rev", "el"),
    )
    variable_translations = {}
    model = "motoneuron"
    conductance_density_parameter = "gl"


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


class SetRate(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval.
    """

    def __init__(self, population_source, population_neuron, interval=20.0):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.interval = interval

    def __call__(self, t):
        rate = 83 + 70 * np.sin(25 * t / 1000)
        self.population_source.set(beta=rate)
        return t + self.interval


class RandomGammaStartSpikeSource(hclass(h.GammaProcess)):
    parameter_names = ("alpha", "beta", "start", "duration")

    def __init__(self, alpha=1, beta=0.1, start=0, duration=0):
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.duration = duration
        self.spike_times = h.Vector(0)
        self.source = self
        self.rec = h.NetCon(self, None)
        self.switch = h.NetCon(None, self)
        self.source_section = None
        self.seed(state.mpi_rank + state.native_rng_baseseed)

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)


class SpikeSourceGammaStart(SpikeSourceGamma):
    translations = build_translations(
        ("alpha", "alpha"),
        ("beta", "beta", 0.001),
        ("start", "start"),
        ("duration", "duration"),
    )
    model = RandomGammaStartSpikeSource

    # Classe SetRate para alterar a taxa de disparo


class SetRate(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval, based on the forces of the muscle units.
    """

    def __init__(
        self, population_source, population_neuron, force_objects, interval=20.0
    ):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects
        self.interval = interval

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        rate = 83 + ((200 - total_force) * 0.01)
        self.population_source.set(beta=rate)

        return t + self.interval


class SetRate(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval, based on the forces of the muscle units.
    """

    def __init__(
        self, population_source, population_neuron, force_objects, interval=20.0, ref=0
    ):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects
        self.interval = interval
        self.ref = ref
        print(f"valor: {self.ref}")

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        rate = 83 + ((self.ref - total_force) * 0.01)
        self.population_source.set(beta=rate)

        return t + self.interval

class SetRateGamma(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval, based on the forces of the muscle units.
    """

    def __init__(
        self, population_source, interval=20.0, rate=100
    ):
        self.population_source = population_source
        self.interval = interval
        self.rate = rate
        print(f"rate: {self.rate}")

    def __call__(self, t):
        
        self.population_source.set(beta=self.rate)

        return t + self.interval



class SetRateIntControl(object):
    """
Um callback que altera a taxa de disparo de uma população de processos
Poisson em intervalos fixos, com base nas forças das unidades motoras.
"""

    def __init__(self, population_source, population_neuron, force_objects, interval=20.0, ref=0, feedforward=0, strength = 1, Kp=0.001, Ki=0.00001, delay=60):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects  
        self.interval = interval
        self.ref = ref
        self.feedforward = feedforward
        self.strength = strength
        self.Kp = Kp
        self.Ki = Ki
        self.Fint = 0
        self.F = dict()
        self.F[20] = 0
        self.F[40] = 0
        self.F[60] = 0
        self.F[80] = 0
        self.F[100] = 0
        self.delay = delay
        
        print(f'valor: {self.ref}')

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        erro = (self.ref - self.F[self.delay])
        self.Fint =  self.Fint + erro*self.interval
        rate = self.feedforward + self.strength*(erro*self.Kp + self.Ki*self.Fint)
        self.population_source.set(beta=rate+0.001)
        self.F[20] = total_force
        self.F[40] = self.F[20]
        self.F[60] = self.F[40]
        self.F[80] = self.F[60]
        self.F[100] = self.F[80]
        
        

        return t + self.interval
