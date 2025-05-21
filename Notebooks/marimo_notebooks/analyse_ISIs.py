import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from collections import defaultdict
    return defaultdict, np, os, pd, plt


@app.cell
def _(np):
    def empty_array():
        return np.array([]).reshape(0,1)
    return (empty_array,)


@app.cell
def _():
    folders = ['P01_iEMG_Ramps_VL_Plateau_Spikes_CSV',
               'P01_iEMG_Ramps_VM_Plateau_Spikes_CSV', 
               'P02_iEMG_Ramps_VL_Plateau_Spikes_CSV',
               'P02_iEMG_Ramps_VM_Plateau_Spikes_CSV',
               'TA_plateau_spikes_csv',
               'FDI_plateau_spikes_csv']
    return (folders,)


@app.cell
def _(defaultdict, empty_array, folders, np, os, pd):
    ISIs = dict()
    muscles = ['VM', 'VL', 'TA', 'FDI']
    for muscle in muscles:
        ISIs[muscle] = dict()


    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            data = pd.read_csv(f'{folder}/{file}', skiprows=6, sep=',')
            muscle = folder.split('_')[3]
            if muscle != 'VM' and muscle != 'VL': 
                muscle = folder.split('_')[0]
            if muscle == 'VM' or muscle == 'VL': force_level = int(file.split('_')[4])
            elif muscle == 'TA': force_level = int(file.split('_')[2][1:])
            else: force_level = 1
            ISIs[muscle][force_level] = defaultdict(empty_array)
            mus = data['MU'].unique()
            for mu in mus:
                data_mu = data.query('MU == @mu')
                plateaus = data_mu['Plateau'].unique()
                for plateau in plateaus:
                    data_plateau = data_mu.query('Plateau == @plateau')
                    ISIs[muscle][force_level][mu] = np.vstack((ISIs[muscle][force_level][mu], np.diff(data_plateau['Spike_Time']).reshape(-1,1)))

    ISI_sd = dict()
    ISI_CV = dict()
    ISI_mean = dict()
    FR_mean = dict()
    for muscle in ISIs.keys():
        ISI_sd[muscle] = dict()
        ISI_CV[muscle] = dict()
        ISI_mean[muscle] = dict()
        FR_mean[muscle] = dict()
        for force_level in ISIs[muscle].keys():
            ISI_sd[muscle][force_level] = np.array([]).reshape(0,1)
            ISI_CV[muscle][force_level] = np.array([]).reshape(0,1)
            ISI_mean[muscle][force_level] = np.array([]).reshape(0,1)
            FR_mean[muscle][force_level] = np.array([]).reshape(0,1)
            for mu in ISIs[muscle][force_level].keys():
                ISI_sd[muscle][force_level] = np.vstack((ISI_sd[muscle][force_level], ISIs[muscle][force_level][mu].std(ddof=1)))
                ISI_mean[muscle][force_level] = np.vstack((ISI_mean[muscle][force_level], ISIs[muscle][force_level][mu].mean()))
                ISI_CV[muscle][force_level] = np.vstack((ISI_CV[muscle][force_level], ISIs[muscle][force_level][mu].std(ddof=1)/ISIs[muscle][force_level][mu].mean()))
                FR_mean[muscle][force_level] = np.vstack((FR_mean[muscle][force_level], (1/ISIs[muscle][force_level][mu]).mean()))

    data_array = np.array([]).reshape(0, 6)
    for muscle in ISI_sd.keys():
        for force_level in ISI_sd[muscle].keys():
            for mu in range(len(ISI_sd[muscle][force_level])):
                data_array = np.vstack((data_array, np.array([[muscle, force_level, ISI_mean[muscle][force_level][mu].squeeze(), 
                                                             ISI_sd[muscle][force_level][mu].squeeze(), ISI_CV[muscle][force_level][mu].squeeze(), 
                                                             FR_mean[muscle][force_level][mu].squeeze()]])))
    df = pd.DataFrame({'Muscle': data_array[:,0], 'Force Level': data_array[:,1], 'ISI mean': data_array[:,2],
                       'ISI SD': data_array[:,3], 'ISI CV': data_array[:,4], 'FR mean': data_array[:,5]})
    df['ISI mean'] = df['ISI mean'].astype(float)
    df['ISI SD'] = df['ISI SD'].astype(float)
    df['ISI CV'] = df['ISI CV'].astype(float)
    df['FR Mean'] = df['FR mean'].astype(float)
    df['Force Level'] = df['Force Level'].astype(int)
    df = df.query('`ISI SD` < 0.4')
    df = df.query('`ISI SD` < 0.02 or `ISI mean`>0.1')
    df.to_csv('ISI_statistics.csv', index=False)
    return ISI_CV, ISI_mean, ISI_sd, df


@app.cell
def _(ISI_CV, ISI_mean, ISI_sd, plt):
    def plotSD_M(muscle_name, force_levels):
        plt.figure()
        for level in force_levels:
            plt.scatter(ISI_mean[muscle_name][level], ISI_sd[muscle_name][level])
        plt.legend(force_levels)
        plt.show()

    def plotFR_CV(muscle_name, force_levels):
        plt.figure()
        for level in force_levels:
            plt.scatter(ISI_CV[muscle_name][level], 1/ISI_mean[muscle_name][level])
        plt.legend(force_levels)
        plt.show()
    return plotFR_CV, plotSD_M


@app.cell
def _(plotFR_CV, plotSD_M):
    plotSD_M('TA', [20, 40])
    plotFR_CV('TA', [20, 40])
    return


@app.cell
def _(plotFR_CV, plotSD_M):
    plotSD_M('VM', [15,30])
    plotFR_CV('VM', [15,30])
    return


@app.cell
def _(plotFR_CV, plotSD_M):
    plotSD_M('VL', [5,15,30])
    plotFR_CV('VL', [5,15,30])
    return


@app.cell
def _(plotFR_CV, plotSD_M):
    plotSD_M('FDI', [1])
    plotFR_CV('FDI', [1])
    return


@app.cell
def _(df, plt):
    data_considered = df.query('`Force Level`>=5')
    plt.scatter(data_considered['ISI mean'], data_considered['ISI SD'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
