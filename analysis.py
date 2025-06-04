import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import sys
    sys.path.append('./../')
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import welch, detrend, butter, filtfilt, csd
    from scipy import stats
    import sympy as sym

    path = 'diabetes/results/'
    batch_name = 'variability'
    trials = np.arange(50)



    trials_long = np.arange(1, 5)

    conditions = ['normal', 'low_affected', 'severe']

    t_start = 4000
    t_end = 10000
    return (
        batch_name,
        butter,
        conditions,
        csd,
        detrend,
        filtfilt,
        np,
        path,
        pd,
        plt,
        stats,
        t_end,
        t_start,
        trials,
        trials_long,
        welch,
    )


@app.cell
def _(butter, filtfilt, np, plt):
    def select_mns_randomly(data, t_start, t_end, size=4, column_spikes=1):
        steady_data = data[(data[:, column_spikes] >= t_start) & (data[:, column_spikes] <= t_end)]
        unique_neurons = np.unique(data[:, 0])
        fr = compute_fr(unique_neurons, steady_data, t_start, t_end, column_spikes=column_spikes)
        # Filtrar dados da fase de estado estacionário

        # Seleção dos neurônios
        selected_neurons = unique_neurons
        # selected_neurons = selected_neurons[np.where((fr < 200))[0]].astype(int)
        selected_neurons = np.random.choice(selected_neurons, size=size)

        return selected_neurons




    def select_mns_regular(data, t_start, t_end, column_spikes=1):
        unique_neurons = np.unique(data[:, 0])

        # Filtrar dados da fase de estado estacionário
        steady_data = data[(data[:, column_spikes] >= t_start) & (data[:, column_spikes] <= t_end)]
        ISI_CV, ISI_mean = compute_cv(unique_neurons, steady_data, t_start, t_end, column_spikes=column_spikes)

        fr = compute_fr(unique_neurons, data, t_start, t_end, column_spikes=column_spikes)

        # Seleção dos neurônios

        selection_criteria = np.where((fr > 5) & (fr < 15) & (ISI_CV <=0.3))[0]
        selected_neurons = unique_neurons[selection_criteria].astype(int)
        fr = fr[selection_criteria]
        mn_number =6#int(min(np.random.randint(low=4, high=12, size=1)[0], len(fr)))
        if len(selected_neurons) > mn_number:
             selected_neurons = selected_neurons[np.argsort(fr)][:mn_number]
        # print(ISI_CV[selection_criteria])
        return selected_neurons

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
            ISI_mean = 0
            ISI_CV = 1
        return ISI_CV, ISI_mean

    def compute_cv(selected_neurons, data, t_start, t_end, column_spikes=1):
        steady_data = data[(data[:, column_spikes] >= t_start) & (data[:, column_spikes] <= t_end)]
        ISI_CV = np.zeros(len(selected_neurons))
        ISI_mean = np.zeros(len(selected_neurons))
        i = 0
        for neuron in selected_neurons:
            ISI_CV[i], ISI_mean[i] = compute_mn_cv(steady_data[steady_data[:, 0] == neuron, column_spikes], t_start=t_start)
            i = i + 1    

        return ISI_CV, ISI_mean


    def plot_mn_fr(mn_rate_mean_mean, mn_rate_mean_CV, conditions, pd, mode):
        import os
        os.makedirs('diabetes', exist_ok=True)
        mean_fr = np.hstack((np.mean(mn_rate_mean_mean[conditions[0]]), 
                             np.mean(mn_rate_mean_mean[conditions[1]]),
                                     np.mean(mn_rate_mean_mean[conditions[2]])))
        sem_fr = np.hstack((mn_rate_mean_mean[conditions[0]].std()/np.sqrt(len(mn_rate_mean_mean[conditions[0]])), 
                               mn_rate_mean_mean[conditions[1]].std()/np.sqrt(len(mn_rate_mean_mean[conditions[1]])),
                               mn_rate_mean_mean[conditions[2]].std(ddof=1)/np.sqrt(len(mn_rate_mean_mean[conditions[2]]))))
        fig, ax = plt.subplots()
        ax.errorbar([1,2,3], mean_fr, fmt='.', yerr=sem_fr, capsize=5, color='black')
        ax.grid()
        ax.scatter(1+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[0]])), mn_rate_mean_mean[conditions[0]])
        ax.scatter(2+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[1]])), mn_rate_mean_mean[conditions[1]])
        ax.scatter(3+0.1*np.random.normal(size=len(mn_rate_mean_mean[conditions[2]])), mn_rate_mean_mean[conditions[2]])
        ax.set_xticks([1,2,3])
        ax.set_xticklabels([conditions[0], conditions[1], conditions[2]])
        ax.set_ylabel('MN firing rate (pps)')
        fig.tight_layout()
        fig.savefig(f'diabetes/mn_firing_rate_comparison_{mode}.png')
        plt.close(fig)
        # Salvar dados em CSV
        for cond in conditions:
            df = pd.DataFrame({
                'firing_rate': mn_rate_mean_mean[cond].flatten(),
                'ISI_CV': mn_rate_mean_CV[cond].flatten()
            })
            df.to_csv(f'diabetes/mn_firing_rate_{cond}_{mode}.csv', index=False)
        df_mean = pd.DataFrame({
            'condition': conditions,
            'mean_firing_rate': mean_fr,
            'sem_firing_rate': sem_fr
        })
        df_mean.to_csv(f'diabetes/mn_firing_rate_summary_{mode}.csv', index=False)

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




    return (
        compute_cv,
        compute_fr,
        firing_rate,
        plot_mn_fr,
        select_mns_randomly,
        select_mns_regular,
    )


@app.cell
def _(
    batch_name,
    compute_cv,
    compute_fr,
    conditions,
    np,
    path,
    plot_mn_fr,
    select_mns_randomly,
    select_mns_regular,
    stats,
    t_end,
    t_start,
):

    def fr_analysis(trials, mode='regular', pd=None):
        force_level = 20
        mn_rate_mean_mean = dict()
        mn_rate_mean_mean[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[2]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV = dict()
        mn_rate_mean_CV[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[2]] = np.array([]).reshape(-1,1)
        force_mean = 0
        CV_mean = 0
        n = 0
        for trial in trials:
            for condition in conditions:
                data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}/cell_spike_ref_{force_level}.csv', delimiter=',')
                force = pd.read_csv(f'{path}force_{condition}_{trial}_{batch_name}/force_ref{force_level}.csv', delimiter=',').values
                data = data.values
                if condition == 'severe':
                    force = force[force[:,0]>t_start,1]
                    force_mean = force_mean + force
                    CV_mean = CV_mean +force.std()/force.mean()
                    n = n + 1
                if mode=='randomly': selected_neurons = select_mns_randomly(data, t_start=t_start, t_end=t_end, size=4)
                if mode=='regular': selected_neurons = select_mns_regular(data, t_start=t_start, t_end=t_end)
                mns_rate_mean = compute_fr(selected_neurons, data, t_start, t_end)
                ISI_CV, _ = compute_cv(selected_neurons, data, t_start, t_end)
                ISI_CV = ISI_CV[mns_rate_mean>=0.01].reshape(-1,1)
                mns_rate_mean = mns_rate_mean[mns_rate_mean>=0.01].reshape(-1,1)        
                mn_rate_mean_mean[condition] = np.vstack((mn_rate_mean_mean[condition], mns_rate_mean))
                mn_rate_mean_CV[condition] = np.vstack((mn_rate_mean_CV[condition], ISI_CV))
        force_mean = force_mean/n
        unique_neurons = np.unique(data[:, 0])        
        ISI_CV_all, _ = compute_cv(unique_neurons, data, t_start, t_end)
        print('Mean force: ', force_mean.mean(), 'CV force:', CV_mean)
        data = np.hstack((data, np.zeros((len(data),1))))
        for i in unique_neurons:
            data[data[:,0]==int(i),2] = ISI_CV_all[unique_neurons==int(i)]
        plot_mn_fr(mn_rate_mean_mean, mn_rate_mean_CV, conditions, pd, mode)
        print(f'FR normal: {mn_rate_mean_mean["normal"].mean()}, FR low_affected: {mn_rate_mean_mean["low_affected"].mean()}, FR severe:                        {mn_rate_mean_mean["severe"].mean()}')
        t_statistic_ind, p_value_ind = stats.ttest_ind(a=mn_rate_mean_mean[conditions[0]], b=mn_rate_mean_mean[conditions[2]])
        print("p-value closed-open:", p_value_ind)
        t_statistic_ind, p_value_ind = stats.ttest_ind(a=mn_rate_mean_mean[conditions[1]], b=mn_rate_mean_mean[conditions[2]])
        print("p-value closed_weak-open:", p_value_ind)
        t_statistic_ind, p_value_ind = stats.ttest_ind(a=mn_rate_mean_mean[conditions[0]], b=mn_rate_mean_mean[conditions[1]])
        print("p-value closed-closed_weak:", p_value_ind)
    return (fr_analysis,)


@app.cell
def _(fr_analysis, pd, trials):
    fr_analysis(trials=trials, mode='randomly', pd=pd)
    return


@app.cell
def _(fr_analysis, pd, trials):
    fr_analysis(trials=trials, mode='regular', pd=pd)
    return


@app.cell
def _(
    batch_name,
    compute_cv,
    conditions,
    np,
    path,
    pd,
    plt,
    select_mns_regular,
    t_end,
    t_start,
):
    def mn_cv_friring_rate(trial):


        force_level = 20

        mn_rate_mean_mean = dict()
        mn_rate_mean_mean[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[2]] = np.array([]).reshape(-1,1)

        mn_rate_mean_CV = dict()
        mn_rate_mean_CV[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[2]] = np.array([]).reshape(-1,1)

        for condition in conditions:
            data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}/cell_spike_ref_{force_level}.csv', delimiter=',')

            data = data.values

            selected_neurons = select_mns_regular(data, t_start=t_start, t_end=t_end)



            unique_neurons = np.unique(data[:, 0])        
            ISI_CV_all, _ = compute_cv(unique_neurons, data, t_start, t_end)

            data = np.hstack((data, np.zeros((len(data),1))))
            for i in unique_neurons:
                data[data[:,0]==int(i),2] = ISI_CV_all[unique_neurons==int(i)]
            plt.figure()
            plt.scatter(data[:,1], data[:,0], c=data[:,2], cmap='Reds', vmin=0, vmax=1)
            selection = np.nonzero(np.in1d(data[:,0], selected_neurons))[0]
            # plt.scatter(data[selection,1], data[selection,0], c=data[selection,2], cmap='Blues', vmin=0, vmax=1)
            plt.title(condition)
            plt.show()
    return (mn_cv_friring_rate,)


@app.cell
def _(mn_cv_friring_rate):
    mn_cv_friring_rate(0)
    return


@app.cell
def _(
    batch_name,
    compute_cv,
    compute_fr,
    conditions,
    np,
    path,
    plt,
    select_mns_randomly,
    t_end,
    t_start,
):
    def fr_cv(trials, pd=None):
        force_level = 20

        mn_rate_mean_mean = dict()
        mn_rate_mean_mean[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_mean[conditions[2]] = np.array([]).reshape(-1,1)

        mn_rate_mean_CV = dict()
        mn_rate_mean_CV[conditions[0]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[1]] = np.array([]).reshape(-1,1)
        mn_rate_mean_CV[conditions[2]] = np.array([]).reshape(-1,1)

        color = dict()
        color[conditions[0]] = 'Blues'
        color[conditions[1]] = 'Oranges'
        color[conditions[2]] = 'Greens'

        neurons_index = dict()
        neurons_index[conditions[0]] = np.array([]).reshape(-1,1)
        neurons_index[conditions[1]] = np.array([]).reshape(-1,1)
        neurons_index[conditions[2]] = np.array([]).reshape(-1,1)




        for trial in trials:
            for condition in conditions:
                data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}/cell_spike_ref_{force_level}.csv', delimiter=',')

                data = data.values

                selected_neurons = select_mns_randomly(data, t_start=t_start, t_end=t_end, size=100)
                mns_rate_mean = compute_fr(selected_neurons, data, t_start, t_end)
                ISI_CV, _ = compute_cv(selected_neurons, data, t_start, t_end)
                ISI_CV = ISI_CV[mns_rate_mean>=0.01].reshape(-1,1)
                selected_neurons = selected_neurons[mns_rate_mean>=0.01].reshape(-1,1)

                # print(ISI_CV)

                mns_rate_mean = mns_rate_mean[mns_rate_mean>=0.01].reshape(-1,1) 
                mn_rate_mean_mean[condition] = np.vstack((mn_rate_mean_mean[condition], mns_rate_mean))
                mn_rate_mean_CV[condition] = np.vstack((mn_rate_mean_CV[condition], ISI_CV))
                neurons_index[condition] = np.vstack((neurons_index[condition], selected_neurons.reshape(-1,1)))

        neurons_index[conditions[0]][neurons_index[conditions[0]]> 120] = 250
        neurons_index[conditions[1]][neurons_index[conditions[1]]> 120] = 250
        neurons_index[conditions[2]][neurons_index[conditions[2]]> 120] = 250

        neurons_index[conditions[0]][(neurons_index[conditions[0]]<= 120) & (neurons_index[conditions[0]]> 60)] = 100
        neurons_index[conditions[1]][(neurons_index[conditions[1]]<= 120) & (neurons_index[conditions[1]]> 60)] = 100
        neurons_index[conditions[2]][(neurons_index[conditions[2]]<= 120) & (neurons_index[conditions[2]]> 60)] = 100        

        neurons_index[conditions[0]][(neurons_index[conditions[0]]< 60)] = 10
        neurons_index[conditions[1]][(neurons_index[conditions[1]]< 60)] = 10
        neurons_index[conditions[2]][(neurons_index[conditions[2]]< 60)] = 10        

        expdata = pd.read_csv('results/ISI_statistics.csv')
        data_muscle = expdata.query('Muscle == "FDI"')

        import matplotlib.patches as mpatches
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        fig, ax = plt.subplots(figsize=(8,6))
        # Plot principal
        s0 = ax.scatter(mn_rate_mean_CV[conditions[0]], mn_rate_mean_mean[conditions[0]], c=neurons_index[conditions[0]], cmap=color[conditions[0]], vmin=1, vmax=250)
        s1 = ax.scatter(mn_rate_mean_CV[conditions[1]], mn_rate_mean_mean[conditions[1]], c=neurons_index[conditions[1]], cmap=color[conditions[1]], vmin=1, vmax=250)
        s2 = ax.scatter(mn_rate_mean_CV[conditions[2]], mn_rate_mean_mean[conditions[2]], c=neurons_index[conditions[2]], cmap=color[conditions[2]], vmin=1, vmax=250)
        # ax.scatter(data_muscle['ISI CV'], 1/data_muscle['ISI mean'], color='m')
        ax.set_xlabel('CV')
        ax.set_ylabel('Firing rate mean (pps)')
        ax.grid(True, linestyle='--', alpha=0.7)
        # Legenda com cores fixas (sem patches)
        legend_labels = [conditions[0], conditions[1], conditions[2]]
        legend_colors = ['blue', 'orange', 'green']
        legend_handles = []
        for legend_color, label in zip(legend_colors, legend_labels):
            legend_handles.append(ax.scatter([], [], color=legend_color, label=label))
        ax.legend(handles=legend_handles, loc='upper left')
        # Inset 1 (zoom1) - subir
        axins1 = fig.add_axes([0.5, 0.72, 0.35, 0.25])  # [left, bottom, width, height] (subiu de 0.6 para 0.72)
        axins1.scatter(mn_rate_mean_CV[conditions[0]], mn_rate_mean_mean[conditions[0]], c=neurons_index[conditions[0]], cmap=color[conditions[0]], vmin=1, vmax=250)
        axins1.scatter(mn_rate_mean_CV[conditions[1]], mn_rate_mean_mean[conditions[1]], c=neurons_index[conditions[1]], cmap=color[conditions[1]], vmin=1, vmax=250)
        axins1.scatter(mn_rate_mean_CV[conditions[2]], mn_rate_mean_mean[conditions[2]], c=neurons_index[conditions[2]], cmap=color[conditions[2]], vmin=1, vmax=250)
        axins1.set_xlim(0, 0.3)
        axins1.set_ylim(5, 15)
        axins1.set_xlabel('CV', fontsize=8)
        axins1.set_ylabel('FR', fontsize=8)
        axins1.tick_params(axis='both', which='major', labelsize=8)
        axins1.grid(True, linestyle='--', alpha=0.7)
        # Inset 2 (zoom2)
        axins2 = fig.add_axes([0.5, 0.32, 0.35, 0.25])
        axins2.scatter(mn_rate_mean_CV[conditions[0]], mn_rate_mean_mean[conditions[0]], c=neurons_index[conditions[0]], cmap=color[conditions[0]], vmin=1, vmax=250)
        axins2.scatter(mn_rate_mean_CV[conditions[1]], mn_rate_mean_mean[conditions[1]], c=neurons_index[conditions[1]], cmap=color[conditions[1]], vmin=1, vmax=250)
        axins2.scatter(mn_rate_mean_CV[conditions[2]], mn_rate_mean_mean[conditions[2]], c=neurons_index[conditions[2]], cmap=color[conditions[2]], vmin=1, vmax=250)
        axins2.set_xlim(0, 0.2)
        axins2.set_ylim(15, 25)
        axins2.set_xlabel('CV', fontsize=8)
        axins2.set_ylabel('FR', fontsize=8)
        axins2.tick_params(axis='both', which='major', labelsize=8)
        axins2.grid(True, linestyle='--', alpha=0.7)
        fig.savefig('diabetes/fr_cv_scatter_full.png', bbox_inches='tight')
        plt.close(fig)
        # Salvar dados em CSV
        import os
        os.makedirs('diabetes', exist_ok=True)
        for cond in conditions:
            df = pd.DataFrame({
                'firing_rate': mn_rate_mean_mean[cond].flatten(),
                'ISI_CV': mn_rate_mean_CV[cond].flatten(),
                'neuron_index': neurons_index[cond].flatten()
            })
            df.to_csv(f'diabetes/fr_cv_{cond}.csv', index=False)
    return (fr_cv,)


@app.cell
def _(fr_cv, pd, trials):
    fr_cv(trials, pd=pd)
    return


@app.cell
def _(
    batch_name,
    conditions,
    np,
    path,
    plt,
    select_mns_randomly,
    select_mns_regular,
):
    def mu_skewness(trials, mode='regular', pd=None):
        import os

        force_level = 20

        ISI = dict()
        ISI[conditions[0]] = np.array([]).reshape(-1,1)
        ISI[conditions[1]] = np.array([]).reshape(-1,1)
        ISI[conditions[2]] = np.array([]).reshape(-1,1)


        for trial in trials:
            for condition in conditions:
                # print(condition)
                data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}/cell_spike_ref_{force_level}.csv', delimiter=',')
                data = data.values
                t_start = 4000
                t_end = 10000
                if mode == 'randomly': selected_neurons = select_mns_randomly(data, t_start=t_start, t_end=t_end)
                if mode == 'regular': selected_neurons = select_mns_regular(data, t_start=t_start, t_end=t_end)
                steady_data = data[(data[:, 1] >= t_start) & (data[:, 1] <= t_end)]
                for neuron in selected_neurons:
                    ISI_mn = np.diff(steady_data[steady_data[:,0] == neuron,1]).reshape(-1,1) 
                    ISI[condition] = np.vstack((ISI[condition], ISI_mn))
        os.makedirs('diabetes', exist_ok=True)
        plt.figure()
        for condition in conditions:        
            plt.hist(ISI[condition], bins=np.arange(0, 300, 10), density=True, alpha=0.5, label=condition)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'diabetes/ISI_hist_{mode}.png')
        plt.close()

        # Salvar dados em CSV

        os.makedirs('diabetes', exist_ok=True)
        for cond in conditions:
            df = pd.DataFrame({
                'ISI': ISI[cond].flatten()
            })
            df.to_csv(f'diabetes/ISI_hist_{cond}_{mode}.csv', index=False)
    return (mu_skewness,)


@app.cell
def _(mu_skewness, pd, trials):
    mu_skewness(trials=trials, mode='randomly', pd=pd)
    return


@app.cell
def _(mu_skewness, pd, trials):
    mu_skewness(trials=trials, mode='regular', pd=pd)
    return


@app.cell
def _(batch_name, conditions, detrend, path, plt, t_start, welch):
    def force_spectrum(trials, pd=None):

        force_level = 20
        delta_t = 0.05


        force = dict()
        n = dict()
        for condition in conditions:
            force[condition] = 0
            n[condition] = 0

        for trial in trials:
            for condition in conditions:
                force_data = pd.read_csv(f'{path}force_{condition}_{trial}_{batch_name}_long/force_ref{force_level}.csv', delimiter=',').values
                force[condition] = force[condition] + force_data[force_data[:,0]> t_start, 1]

        Pforce = dict()
        plt.figure()

        nperseg = 176000

        for condition in conditions:
            force[condition] = force[condition]/len(trials)
            f, Pforce[condition] = welch(detrend(force[condition]), fs=1/0.00005, nperseg=nperseg, nfft=nperseg, detrend=False)
            plt.plot(f, (Pforce[condition])/(Pforce[condition]).max(), label=condition)

        plt.legend([conditions[0], conditions[1], conditions[2]])
        plt.xlim(0, 5)
        plt.title('Force')
        plt.tight_layout()
        plt.savefig('diabetes/force_spectrum.png')
        plt.close()

        # Salvar dados em CSV
        import os
        os.makedirs('diabetes', exist_ok=True)
        for cond in conditions:
            df = pd.DataFrame({
                'frequency': f,
                'normalized_power': (Pforce[cond])/(Pforce[cond]).max()
            })
            df.to_csv(f'diabetes/force_spectrum_{cond}.csv', index=False)
    return (force_spectrum,)


@app.cell
def _(force_spectrum, pd, trials_long):
    force_spectrum(trials=trials_long, pd=pd)
    return


@app.cell
def _(
    batch_name,
    conditions,
    csd,
    detrend,
    firing_rate,
    np,
    path,
    plt,
    select_mns_randomly,
    select_mns_regular,
    stats,
    t_end,
    t_start,
    welch,
):
    def coherence_mu(trials, mode='regular', pd=None):
        force_level = 20

        delta_t = 0.05
        divisions = 2
        nperseg = 176000//divisions

        Prate = dict()
        Pfr = dict()
        Prate_fr = dict()
        n = dict()
        for condition in conditions:
            Prate[condition]= 0
            Pfr[condition] = 0
            Prate_fr[condition] = 0
            n[condition] = 0

        force_feedback_delay = 60


        for trial in trials:
            for condition in conditions:
                data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}_long/cell_spike_ref_{force_level}.csv', delimiter=',')
                source_data = pd.read_csv(f'{path}spikefeedbacksource_{condition}_{trial}_{batch_name}_long/spike_data_ref_{force_level}.csv', 
                                          delimiter=',')
                data = data.values
                source_data['spike_time'] = source_data['spike_time'].str.replace(' ms', '')
                source_data['spike_time'] = source_data['spike_time'].astype('float')
                source_data = source_data.values
                steady_data = data[(data[:, 1] >= t_start) & (data[:, 1] <= 180000)]
                steady_source_data = source_data[(source_data[:, 1] >= t_start) & (source_data[:, 1] <= 180000)]
                if mode == 'randomly': selected_neurons = select_mns_randomly(data, t_start=t_start, t_end=t_end, size=1)
                if mode == 'regular': selected_neurons = select_mns_regular(data, t_start=t_start, t_end=t_end)
                rate = 0
                dc_neurons = np.unique(steady_source_data[:,0])
                np.random.shuffle(dc_neurons)
                for dc in dc_neurons[:50]:
                    fr_source, t = firing_rate(steady_source_data[steady_source_data[:,0]==dc,1]-t_start, 
                                               delta_t=0.00005, filtro_ordem=4, freq_corte=500, tempo_max=(180000-t_start)/1000)
                    rate = rate + fr_source
                rate = rate/50
                for neuron in selected_neurons:
                    fr_new, t =  firing_rate(steady_data[steady_data[:,0]==neuron,1]-t_start, delta_t=0.00005,
                                             filtro_ordem=4, freq_corte=5, tempo_max=(180000-t_start)/1000)
                    f, Sfr = welch(detrend(fr_new), fs=1/0.00005, nperseg=nperseg, detrend=None)
                    Pfr[condition] = Pfr[condition] + Sfr    

                    f, Srate = welch(detrend(rate), fs=1/0.00005, nperseg=nperseg, detrend=None)
                    Prate[condition] = Prate[condition] + Srate
                    f, Sratefr = csd(detrend(rate), detrend(fr_new), fs=1/0.00005, nperseg=nperseg, detrend=None)
                    Prate_fr[condition] = Prate_fr[condition] + Sratefr
                    n[condition] = n[condition] + 1

        Coh = dict()
        for condition in conditions:
            Pfr[condition] = Pfr[condition]/n[condition]
            Prate[condition] = Prate[condition]/n[condition]
            Prate_fr[condition] = Prate_fr[condition]/n[condition]       
            Coh[condition] = np.abs(Prate_fr[condition])**2/(Prate[condition]*Pfr[condition]+1e-6)

            alpha = 0.05
            k = divisions*n[condition]
            cl = stats.f.pdf(1-alpha, 2, 2*(k-1))/(k-1+stats.f.pdf(0.95, 2, 2*(k-1)))

            plt.figure()
            plt.plot(f, Coh[condition])
            plt.hlines(cl, 0, 5, color='k')
            plt.title(condition)
            plt.xlim(f[0], 5)
            plt.tight_layout()
            plt.savefig(f'diabetes/coherence_mu_{condition}_{mode}.png')
            plt.close()

        # Salvar dados em CSV
        import os
        os.makedirs('diabetes', exist_ok=True)
        for cond in conditions:
            df = pd.DataFrame({
                'frequency': f,
                'coherence': Coh[cond]
            })
            df.to_csv(f'diabetes/coherence_mu_{cond}_{mode}.csv', index=False)
    return (coherence_mu,)


@app.cell
def _(coherence_mu, pd, trials_long):
    coherence_mu(trials=trials_long, mode='randomly', pd=pd)
    return


@app.cell
def _(coherence_mu, pd, trials_long):
    coherence_mu(trials=trials_long, mode='regular', pd=pd)
    return


@app.cell
def _(
    batch_name,
    conditions,
    detrend,
    firing_rate,
    path,
    plt,
    select_mns_randomly,
    select_mns_regular,
    trials_long,
    welch,
):
    def fr_spectra(trials, mode='regular', pd=None):
        import os
        force_level = 20
        t_start = 4000
        t_end = 180000
        delta_t = 0.05
        nperseg = 176000

        fr_low = dict()
        fr_medium = dict()
        fr_high = dict()
        n_low = dict()
        n_medium = dict()
        n_high = dict()
        Pfr_low = dict()
        Pfr_medium = dict()
        Pfr_high = dict()

        for condition in conditions:
            fr_low[condition] = 0
            fr_medium[condition] = 0
            fr_high[condition] = 0
            n_low[condition] = 0
            n_medium[condition] = 0
            n_high[condition] = 0

        for trial in trials_long:
            for condition in conditions:
                data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}_long/cell_spike_ref_{force_level}.csv', delimiter=',')
                data = data.values
                steady_data = data[(data[:, 1] >= t_start) & (data[:, 1] <= t_end)]
                if mode == 'randomly': selected_neurons = select_mns_randomly(data, t_start=t_start, t_end=10000, size=100)
                if mode == 'regular': selected_neurons = select_mns_regular(data, t_start=t_start, t_end=10000)
                for neuron in selected_neurons:
                    fr_new, t=  firing_rate(steady_data[steady_data[:,0]==neuron,1]-t_start, delta_t=0.00005, filtro_ordem=4, freq_corte=100, tempo_max=(t_end-t_start)/1000)
                    if fr_new.mean() > 0.1:
                        if neuron <= 85:
                            fr_low[condition] = fr_low[condition] + fr_new
                            n_low[condition] = n_low[condition] + 1
                        elif neuron <= 110:
                            fr_medium[condition] = fr_medium[condition] + fr_new
                            n_medium[condition] = n_medium[condition] + 1
                        else:
                            fr_high[condition] = fr_high[condition] + fr_new
                            n_high[condition] = n_high[condition] + 1

        for condition in conditions:
            fr_low[condition] = fr_low[condition]/n_low[condition]
            fr_medium[condition] = fr_medium[condition]/n_medium[condition]
            fr_high[condition] = fr_high[condition]/n_high[condition]
            f, Pfr_low[condition] = welch(detrend(fr_low[condition]), fs=1/0.00005, nperseg=nperseg, nfft = nperseg, detrend=False)
            f, Pfr_medium[condition] = welch(detrend(fr_medium[condition]), fs=1/0.00005, nperseg=nperseg, nfft = nperseg, detrend=False)
            f, Pfr_high[condition] = welch(detrend(fr_high[condition]), fs=1/0.00005, nperseg=nperseg, nfft = nperseg, detrend=False)

        os.makedirs('diabetes', exist_ok=True)
        # Plot all conditions together for each type
        # LOW
        plt.figure()
        for condition in conditions:
            plt.plot(f, Pfr_low[condition], label=condition)
        plt.title(f'Small motor units - {mode}')
        plt.xlim(f[0], 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'diabetes/fr_spectra_low_{mode}_all_conditions.png')
        plt.close()
        # MEDIUM
        plt.figure()
        for condition in conditions:
            plt.plot(f, Pfr_medium[condition], label=condition)
        plt.title(f'Medium motor units - {mode}')
        plt.xlim(f[0], 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'diabetes/fr_spectra_medium_{mode}_all_conditions.png')
        plt.close()
        # HIGH
        plt.figure()
        for condition in conditions:
            plt.plot(f, Pfr_high[condition], label=condition)
        plt.title(f'Big motor units - {mode}')
        plt.xlim(f[0], 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'diabetes/fr_spectra_high_{mode}_all_conditions.png')
        plt.close()
        # Salvar dados em CSV
        for cond in conditions:
            df_low = pd.DataFrame({'frequency': f, 'power': Pfr_low[cond]})
            df_medium = pd.DataFrame({'frequency': f, 'power': Pfr_medium[cond]})
            df_high = pd.DataFrame({'frequency': f, 'power': Pfr_high[cond]})
            df_low.to_csv(f'diabetes/fr_spectra_low_{cond}_{mode}.csv', index=False)
            df_medium.to_csv(f'diabetes/fr_spectra_medium_{cond}_{mode}.csv', index=False)
            df_high.to_csv(f'diabetes/fr_spectra_high_{cond}_{mode}.csv', index=False)
    return (fr_spectra,)


@app.cell
def _(fr_spectra, pd, trials_long):
    fr_spectra(trials_long, mode='randomly', pd=pd)
    return


@app.cell
def _(fr_spectra, pd, trials_long):
    fr_spectra(trials_long, mode='regular', pd=pd)
    return


@app.cell
def _(batch_name, conditions, np, path, pd, plt, select_mns_regular):
    def what_mn_selected(trial):
        import os
        force_level = 20
        for condition in conditions:
            data = pd.read_csv(f'{path}spikedata_{condition}_{trial}_{batch_name}/cell_spike_ref_{force_level}.csv', delimiter=',')
            data = data.values
            t_start = 4000
            t_end = 10000
            selected_neurons = select_mns_regular(data, t_start=t_start, t_end=t_end)
            os.makedirs('diabetes', exist_ok=True)
            fig, ax = plt.subplots()
            ax.plot(data[:,1], data[:,0], '.y')
            ax.plot(data[np.nonzero(np.in1d(data[:,0], selected_neurons))[0],1], data[np.nonzero(np.in1d(data[:,0], selected_neurons))[0],0], '.b')
            ax.set_title(condition)
            fig.tight_layout()
            fig.savefig(f'diabetes/what_mn_selected_{condition}_{trial}.png')
            plt.close(fig)
            # plt.show()  # Removido para evitar exibição interativa
    return (what_mn_selected,)


@app.cell
def _(what_mn_selected):
    what_mn_selected(30)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
