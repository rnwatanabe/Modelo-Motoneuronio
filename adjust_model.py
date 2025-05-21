import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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

    return (
        Classes,
        IonicSpecies,
        NumpyRNG,
        centre,
        functions,
        h,
        neuromuscular_system,
        np,
        os,
        pd,
        plt,
        sim,
        sum_force,
        uniform,
    )


@app.cell
def _(butter, filtfilt, np, plt):
    def compute_fr(selected_neurons, data, t_start, t_end, column_spikes=1):
        window_duration = (t_end - t_start) / 1000  # s
        steady_data = data[(data[:, column_spikes] >= t_start) & (data[:, column_spikes] <= t_end)]
        firing_rates = np.zeros(len(selected_neurons))
        i = 0
        for neuron in selected_neurons:
            n_spikes = np.sum(steady_data[:, 0] == neuron)
            fr = n_spikes / window_duration
            firing_rates[i] = fr
            i = i + 1
        return firing_rates

    def compute_mn_cv(spike_times, t_start):
        ISI = np.diff(spike_times[spike_times> t_start])
        if len(ISI)>3:
            ISI_SD = ISI.std(ddof=1)
            ISI_mean = ISI.mean()
            ISI_CV = ISI_SD/ISI_mean
        else:
            ISI_mean = -1
            ISI_CV = 1
            ISI_SD = -1
        return ISI_CV, ISI_mean, ISI_SD

    def compute_cv(selected_neurons, data, t_start, t_end, column_spikes=1):
        steady_data = data[(data[:, column_spikes] >= t_start) & (data[:, column_spikes] <= t_end)]
        ISI_CV = np.zeros(len(selected_neurons))
        ISI_mean = np.zeros(len(selected_neurons))
        ISI_SD = np.zeros(len(selected_neurons))
        i = 0
        for neuron in selected_neurons:
            ISI_CV[i], ISI_mean[i], ISI_SD[i] = compute_mn_cv(steady_data[steady_data[:, 0] == neuron, column_spikes], t_start=t_start)
            i = i + 1    

        return ISI_CV, ISI_mean, ISI_SD


    def plot_mn_fr(mn_rate_mean_mean, mn_rate_mean_CV, conditions):
        mean_fr = np.hstack((np.mean(mn_rate_mean_mean[conditions[0]]), 
                             np.mean(mn_rate_mean_mean[conditions[1]]),
                                     np.mean(mn_rate_mean_mean[conditions[2]])))
        sem_fr = np.hstack((mn_rate_mean_mean[conditions[0]].std()/np.sqrt(len(mn_rate_mean_mean[conditions[0]])), 
                               mn_rate_mean_mean[conditions[1]].std()/np.sqrt(len(mn_rate_mean_mean[conditions[1]])),
                               mn_rate_mean_mean[conditions[2]].std(ddof=1)/np.sqrt(len(mn_rate_mean_mean[conditions[2]]))))
        plt.errorbar([1,2,3], mean_fr, fmt='.', yerr=sem_fr, capsize=5, color='black')
        plt.grid()
        plt.scatter(1+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[0]])), mn_rate_mean_mean[conditions[0]])#, c=mn_rate_mean_CV[conditions[0]], cmap='Reds', vmin=0, vmax=1)
        plt.scatter(2+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[1]])), mn_rate_mean_mean[conditions[1]])#, c=mn_rate_mean_CV[conditions[1]], cmap='Reds', vmin=0, vmax=1)
        plt.scatter(3+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[2]])), mn_rate_mean_mean[conditions[2]])#, c=mn_rate_mean_CV[conditions[2]], cmap='Reds', vmin=0, vmax=1)
        plt.xticks([1,2,3],[conditions[0], conditions[1], conditions[2]])
        plt.ylabel('MN firing rate (Hz)')

    def firing_rate(spiketrains, delta_t=0.00005, filtro_ordem=4, freq_corte=0.001, tempo_max=1000):
        """
        Função que gera o impulso de Dirac para os tempos de disparo de um neurônio.

        Parâmetros:
            spiketrains: Lista com os trens de disparo de neurônios.
            neuronio: Índice do neurônio a ser processado.
            delta_t: Intervalo de tempo. 
            filtro_ordem : Ordem do filtro Butterworth. 
            freq_corte: Frequência de corte normalizada para o filtro Butterworth.
            tempo_max: Tempo máximo para o eixo x (em milissegundos). 
        """

        # Criação do vetor de tempo
        t = np.arange(0, tempo_max, delta_t)
        fr = np.zeros_like(t)

        # Adiciona o impulso de Dirac em cada tempo de disparo do neurônio
        idx = np.searchsorted(t, spiketrains/1000)
        idx = idx[idx<len(fr)]
        fr[idx] = 1/delta_t
        # Filtro Butterworth
        fs = 1/delta_t
        b, a = butter(filtro_ordem, freq_corte/(fs/2))

        # Aplicação do filtro
        fr = filtfilt(b, a, fr)
        fr[fr<0] = 0
        return fr, t

    return (compute_cv,)


@app.cell
def _(functions):
    functions.update_mod_files()
    return


@app.cell
def _(mo):
    mo.md("""## Model 1""")
    return


@app.cell
def _(
    Classes,
    IonicSpecies,
    NumpyRNG,
    centre,
    functions,
    h,
    neuromuscular_system,
    np,
    os,
    pd,
    plt,
    sim,
    sum_force,
    uniform,
):
    MVC = 500
    force_level = 20

    Kp=0.07
    Ki=0.007
    delay = 60
    connection_prob = 0.1
    Tf = 10000
    mn_number = 250
    CV = 0.01

    for trial in range(1,2):
        rng1 = np.random.default_rng(seed=trial+100)
        timestep = 0.05
        sim.setup(timestep=timestep)

        somas = functions.create_somas(mn_number, diameter_min=77.5, diameter_max=82.5, y_min=18.0, y_max=36.0, seed=trial+500, CV=CV)
        dends = functions.create_dends(mn_number,somas, diameter_min=41.5, diameter_max=62.5, y_min=18.0, y_max=36.0, x_min=-5500, x_max=-6789, seed=trial+500, CV=CV)
        vt = -70 + 12.35*np.exp(np.arange(mn_number)/(mn_number-1)*np.log(20.9/12.35))*(1+CV*rng1.normal(size=mn_number))
        cell_type = Classes.cell_class(
            morphology= functions.soma_dend(somas, dends) ,
            cm=1,    # mF / cm**2
            Ra=0.070, # ohm.mm
            ionic_species={"na": IonicSpecies("na", reversal_potential=50),
                           "ks": IonicSpecies("ks", reversal_potential=-80),
                           "kf": IonicSpecies("kf", reversal_potential=-80)
                           },
            pas_soma = {"conductance_density": functions.create_cond(mn_number, 7e-4, 7e-4, 'soma',  seed=trial+600, CV=10*CV), "e_rev":-70}, #
            pas_dend = {"conductance_density": functions.create_cond(mn_number, 7e-4, 7e-4, 'dendrite',  seed=trial+600, CV=10*CV), "e_rev":-70}, #
            na = {"conductance_density": uniform('soma', 30), "vt":list(vt)},
            kf = {"conductance_density": functions.create_cond(mn_number, 4, 0.5, 'soma',  seed=trial+700, CV=CV), "vt":list(vt)},
            ks = {"conductance_density": uniform('soma', 0.1), "vt":list(vt)},
            syn={"locations": centre('dendrite'),
                "e_syn": 0,
                "tau_syn": 0.6},  
        )

        cells = sim.Population(mn_number, cell_type, initial_values={'v': list(-70*np.ones(mn_number))})

        muscle_units, force_objects, neuromuscular_junctions = neuromuscular_system(cells, mn_number, h, seed=trial+1, 
                                                                                    Fmin=0.02, Fmax=4, Tcmin=110, Tcmax=25,
                                                                                    vel_min=44, vel_max=53, CV=CV)
        # 
        spike_source = sim.Population(400, Classes.SpikeSourceGammaStart(alpha=1))
        feedback_source = sim.Population(400, Classes.SpikeSourceGammaStart(alpha=1)) 
                                                                #start=RandomDistribution('uniform', [0, 3.0], rng=NumpyRNG(seed=4242))))
        syn = sim.StaticSynapse(weight=0.6, delay=0.2)
        syn1 = sim.StaticSynapse(weight=0.6, delay=0.2)
        # nmj = sim.StaticSynapse(weight=1, delay=0.2)
        input_conns = sim.Projection(spike_source, cells, 
                                    sim.FixedProbabilityConnector(connection_prob, location_selector='dendrite', rng=NumpyRNG(seed=trial+200)), 
                                    syn, receptor_type="syn")
        feedback_conns = sim.Projection(feedback_source, cells, 
                                    sim.FixedProbabilityConnector(connection_prob, location_selector='dendrite', rng=NumpyRNG(seed=trial+300)), 
                                    syn1, receptor_type="syn")
        spike_source.record('spikes')
        feedback_source.record('spikes')
        cells.record('spikes')
        # cells[0:2].record_gsyn()
        # cells[0:2].record(('na.m', 'na.h'), locations='soma')
        # cells[0:2].record(('kf.n'), locations='soma')
        # cells[0:2].record(('ks.p'), locations='soma')
        f = dict()
        for mn in range(mn_number):
            f[mn] = h.Vector().record(force_objects[mn]._ref_F)

        refs = np.array([0.2])*MVC

        n = 0
        j = 0
        for ref in refs:   
            spike_source.set(alpha=8)
            feedback_source.set(alpha=20)
            feedforward = 0
            # spike_source.set(beta=feedforward+1e-6)    
            sim.run(Tf, callbacks=[Classes.SetRateIntControl(feedback_source, cells, force_objects, ref=ref, feedforward=0*feedforward, strength=1, Kp=Kp, Ki=Ki, delay=delay),
                                   Classes.SetRateIntControl(spike_source, cells, force_objects, ref=ref, feedforward=feedforward, strength=0, Kp=Kp, Ki=Ki, delay=delay)])
            print('fim simulação')


            force = sum_force(force_objects, h, f)#.as_numpy()

            rate = feedforward + 1*((ref - force) * Kp + Ki*np.cumsum(ref - force)*0.05)

            t = np.arange(0, len(force))*timestep
            print(rate[t>4000].mean())

            print('CV = ', np.std(force.as_numpy()[t>4000])/force.as_numpy()[t>4000].mean())
            plt.plot(t, force)
            plt.hlines(MVC*force_level/100, 0, Tf)
            # plt.ylim(1000, 1200)
            plt.grid()
            plt.show()

            new_folder = f"results/force_adjust"
            os.makedirs(new_folder, exist_ok=True)
            df = pd.DataFrame({'time': t, 'force': force, 'rate': rate})
            # for i in range(100):
            #     df[f'F{str(i)}'] = f[i].as_numpy()
            filename = os.path.join(new_folder, f'force_ref{int(ref/MVC*100)}.csv')
            df.to_csv(filename, index=False) 

            data_source = spike_source.get_data().segments[n]
            data = cells.get_data().segments[n]

            #teste spike_datasource
            spike_df = pd.DataFrame([{"neuron_id": neuron_id, "spike_time": spike_time}
                for neuron_id, spikes in enumerate(data_source.spiketrains)
                for spike_time in spikes])

            new_folder1 = f"results/spikedatasource_adjust"
            os.makedirs(new_folder1, exist_ok=True)    
            filename = os.path.join(new_folder1, f'spike_data_ref_{int(ref/MVC*100)}.csv')
            spike_df.to_csv(filename, index=False)


            feedback_source = feedback_source.get_data().segments[n]


            #teste spike_datasource
            spike_df = pd.DataFrame([{"neuron_id": neuron_id, "spike_time": spike_time}
                for neuron_id, spikes in enumerate(feedback_source.spiketrains)
                for spike_time in spikes])

            new_folder1 = f"results/spikefeedbacksource_adjust"
            os.makedirs(new_folder1, exist_ok=True)    
            filename = os.path.join(new_folder1, f'spike_data_ref_{int(ref/MVC*100)}.csv')
            spike_df.to_csv(filename, index=False)

            # plt.scatter(spike_df["spike_time"], spike_df["neuron_id"], s=4, label=f"ref={ref}")

            # plt.show()

            #teste spike_data
            cell_spike_df = pd.DataFrame([{"neuron_id": neuron_id, "spike_time": spike_time}
                for neuron_id, spikes in enumerate(data.spiketrains)
                for spike_time in spikes])            


            new_folder2 = f"results/spikedata_adjust"
            os.makedirs(new_folder2, exist_ok=True)    
            filename = os.path.join(new_folder2, f'cell_spike_ref_{int(ref/MVC*100)}.csv')
            cell_spike_df.to_csv(filename, index=False)

            data_spikes = pd.read_csv(filename, delimiter=',')
            data_spikes['spike_time'] = data_spikes['spike_time'].str.replace(' ms', '')
            data_spikes['spike_time'] = data_spikes['spike_time'].astype('float')
            data_spikes['spike_time_motor_plate'] = data_spikes['spike_time'].values                
            data_spikes_values = data_spikes.values
            for mn in range(mn_number):                   
                data_spikes_values[data_spikes_values[:,0]==mn,2] = data_spikes_values[data_spikes_values[:,0]==mn,1] + neuromuscular_junctions[mn].delay 
            data_spikes['spike_time_motor_plate'] = data_spikes_values[:,2]
            data_spikes.to_csv(filename, index=False)
            plt.figure
            plt.scatter(cell_spike_df["spike_time"], cell_spike_df["neuron_id"], s=4,label=f"ref={ref}")
            plt.show()
            # plt.figure()
            # n_test = 50
            # plt.plot(t, f[n_test].as_numpy())
            # plt.plot(data_spikes_values[data_spikes_values[:,0]==n_test,1], 0*data_spikes_values[data_spikes_values[:,0]==n_test,0],'.')
            # plt.plot(data_spikes_values[data_spikes_values[:,0]==n_test,2], 0*data_spikes_values[data_spikes_values[:,0]==n_test,0],'.r')
            # plt.show()


        #     Figure(
        #     Panel(vm, ylabel="Membrane potential (mV)", xticks=True, yticks=True),
        # )
        #     plt.show()
            sim.reset()
            for mn in range(mn_number):
                force_objects[mn].F = 0
                force_objects[mn].x1 = 0
                force_objects[mn].x2 = 0
                force_objects[mn].spike = 0
                force_objects[mn].CaSR = 0.0025
                force_objects[mn].CaSRCS = 0.0	
                force_objects[mn].Ca = 1e-10
                force_objects[mn].CaT = 0.0	
                force_objects[mn].AM = 0.0
                force_objects[mn].CaB = 0.0
                force_objects[mn].c1 = 0.154
                force_objects[mn].c2 = 0.11
            n = n + 1
            j = j + 1
        sim.end()


    return cell_spike_df, data_spikes, force_objects, mn_number, ref


@app.cell
def _(data_spikes):
    data_spikes 
    return


@app.cell
def _(compute_cv, data_spikes, mn_number, np):
    t_start = 4000
    t_end = 10000
    ISI_CV, ISI_mean, ISI_SD = compute_cv(np.arange(mn_number), data_spikes.values, t_start, t_end, column_spikes=2)
    return ISI_CV, ISI_SD, ISI_mean


@app.cell
def _(pd):
    expdata = pd.read_csv('results/ISI_statistics.csv')
    data_muscle = expdata.query('Muscle == "VL"')
    return (data_muscle,)


@app.cell
def _(ISI_SD, ISI_mean, data_muscle, plt):
    plt.figure()
    plt.scatter(ISI_mean/1000, ISI_SD/1000, color='y', alpha=0.5)
    plt.scatter(data_muscle['ISI mean'], data_muscle['ISI SD'], alpha=0.2)
    plt.legend(['model', 'experimental'])
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.xlabel('ISI mean (s)')
    plt.ylabel('ISI SD (s)')
    plt.show()
    return


@app.cell
def _(ISI_mean, np):
    np.max(1000/ISI_mean)
    return


@app.cell
def _(ISI_CV, ISI_mean, data_muscle, plt):

    plt.figure()
    plt.scatter(ISI_CV, 1000/ISI_mean, color='y', alpha=0.5)
    plt.scatter(data_muscle['ISI CV'], 1/data_muscle['ISI mean'], alpha=0.2)

    plt.ylim(0,35)
    plt.xlabel('ISI CV')
    plt.ylabel('mean firing rate (pps)')
    plt.legend(['model', 'experimental'])
    plt.show()
    return


@app.cell
def _(ISI_CV, ISI_mean, data_muscle, plt):
    plt.figure()
    plt.scatter(ISI_CV, ISI_mean/1000, color='y',  alpha=0.5)
    plt.scatter(data_muscle['ISI CV'], data_muscle['ISI mean'], alpha=0.2)
    plt.legend(['model', 'experimental'])
    plt.xlabel('ISI CV')
    plt.ylabel('ISI mean (s)')
    plt.ylim(0,1)
    plt.show()
    return


@app.cell
def _(ISI_mean, np):
    np.where((ISI_mean/1000 <= 0.2) & (ISI_mean/1000 >= 0.09))[0]
    return


@app.cell
def _(data_muscle):
    data_muscle['ISI mean'].min()
    return


@app.cell
def _(force_objects):
    force_objects[33].Fmax
    return


@app.cell
def _(cell_spike_df, plt, ref):
    plt.scatter(cell_spike_df["spike_time"], cell_spike_df["neuron_id"], s=4,label=f"ref={ref}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Model 2""")
    return


@app.cell
def _(mn_number, np, plt):
    gmin = 110
    gmax = 25
    Tc = (gmin*np.exp(np.arange(mn_number)/(mn_number-1)*np.log(gmax/gmin)))
    plt.scatter(np.arange(mn_number),  Tc)
    return gmax, gmin


@app.cell
def _(gmax, gmin, mn_number, np):
    np.mean(gmin*np.exp(np.arange(mn_number)/(mn_number-1)*np.log(gmax/gmin)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
