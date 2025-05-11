import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    sys.path.append("./../")
    from src.functions import plot_disparos_neuronios

    _data = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=2000/Ca_cell_spike_ref_440_Umax=2000.csv",
        delimiter=",",
    )
    _data["spike_time"] = _data["spike_time"].str.replace(" ms", "").astype("float")
    _data = _data.values
    _t_start, _t_end = (4500, 54500)
    _window_duration = (_t_end - _t_start) / 1000
    _steady_data = _data[(_data[:, 1] >= _t_start) & (_data[:, 1] <= _t_end)]
    _unique_neurons = np.unique(_data[:, 0])
    _selected_neurons = np.random.choice(_unique_neurons, size=59)
    _firing_rates = []
    for _neuron in _selected_neurons:
        _n_spikes = np.sum(_steady_data[:, 0] == _neuron)
        _fr = _n_spikes / _window_duration
        _firing_rates.append(_fr)
    _x_pos = 0
    _jitter = np.random.uniform(-0.05, 0.05, size=len(_firing_rates))
    plt.figure(figsize=(5, 5))
    plt.scatter(
        np.full(len(_firing_rates), _x_pos) + _jitter,
        _firing_rates,
        color="royalblue",
        alpha=0.8,
        s=60,
    )
    _mean_fr = np.mean(_firing_rates)
    _sem_fr = np.std(_firing_rates) / np.sqrt(len(_firing_rates))
    plt.plot(_x_pos, _mean_fr, "-", color="black", markersize=8)
    plt.errorbar(_x_pos, _mean_fr, yerr=_sem_fr, color="black", capsize=5)
    plt.xticks([_x_pos], ["40% MVC - Controle"])
    plt.ylabel("Taxa de descarga (pps)")
    plt.xlim(-0.3, 0.3)
    plt.tight_layout()
    plt.show()
    return np, pd, plt, sys


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=1600/Ca_cell_spike_ref_220_Umax=1600.csv",
        delimiter=",",
    )
    _data["spike_time"] = _data["spike_time"].str.replace(" ms", "").astype("float")
    _data = _data.values
    _t_start, _t_end = (4500, 54500)
    _window_duration = (_t_end - _t_start) / 1000
    _steady_data = _data[(_data[:, 1] >= _t_start) & (_data[:, 1] <= _t_end)]
    _unique_neurons = np.unique(_data[:, 0])
    _selected_neurons = np.random.choice(_unique_neurons, size=90)
    _firing_rates = []
    for _neuron in _selected_neurons:
        _n_spikes = np.sum(_steady_data[:, 0] == _neuron)
        _fr = _n_spikes / _window_duration
        _firing_rates.append(_fr)
    _x_pos = 0
    _jitter = np.random.uniform(-0.05, 0.05, size=len(_firing_rates))
    plt.figure(figsize=(5, 5))
    plt.scatter(
        np.full(len(_firing_rates), _x_pos) + _jitter,
        _firing_rates,
        color="royalblue",
        alpha=0.8,
        s=60,
    )
    _mean_fr = np.mean(_firing_rates)
    _sem_fr = np.std(_firing_rates) / np.sqrt(len(_firing_rates))
    plt.plot(_x_pos, _mean_fr, "-", color="black", markersize=8)
    plt.errorbar(_x_pos, _mean_fr, yerr=_sem_fr, color="black", capsize=5)
    plt.xticks([_x_pos], ["20% MVC - Controle"])
    plt.ylabel("Taxa de descarga (pps)")
    plt.xlim(-0.3, 0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pd, plt, sys):
    sys.path.append("./../")
    _data = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=1600/Ca_cell_spike_ref_220_Umax=1600.csv",
        delimiter=",",
    )
    _data["spike_time"] = _data["spike_time"].str.replace(" ms", "").astype("float")
    _data = _data.values
    _data1 = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=1600/Ca_cell_spike_ref_440_Umax=1600.csv",
        delimiter=",",
    )
    _data1["spike_time"] = _data1["spike_time"].str.replace(" ms", "").astype("float")
    _data1 = _data1.values
    _t_start_20, _t_end_20 = (4500, 54500)
    _window_20 = (_t_end_20 - _t_start_20) / 1000
    _steady_data_20 = _data[(_data[:, 1] >= _t_start_20) & (_data[:, 1] <= _t_end_20)]
    _t_start_40, _t_end_40 = (4500, 54500)
    _window_40 = (_t_end_40 - _t_start_40) / 1000
    _steady_data_40 = _data1[
        (_data1[:, 1] >= _t_start_40) & (_data1[:, 1] <= _t_end_40)
    ]
    _unique_neurons = np.unique(_data[:, 0])
    _neurons_20 = np.random.choice(_unique_neurons, size=90)
    _neurons_40 = np.random.choice(_unique_neurons, size=59)
    _fr_20 = []
    for _neuron in _neurons_20:
        _n_spikes = np.sum(_steady_data_20[:, 0] == _neuron)
        _fr = _n_spikes / _window_20
        _fr_20.append(_fr)
    _fr_40 = []
    for _neuron in _neurons_40:
        _n_spikes = np.sum(_steady_data_40[:, 0] == _neuron)
        _fr = _n_spikes / _window_40
        _fr_40.append(_fr)
    plt.figure(figsize=(6, 5))
    _x_20 = 0
    _x_40 = 0.7
    _jitter_20 = np.random.uniform(-0.05, 0.05, size=len(_fr_20))
    _jitter_40 = np.random.uniform(-0.05, 0.05, size=len(_fr_40))
    plt.scatter(
        np.full(len(_fr_20), _x_20) + _jitter_20,
        _fr_20,
        color="royalblue",
        alpha=0.8,
        s=60,
        label="20% MVC",
    )
    plt.scatter(
        np.full(len(_fr_40), _x_40) + _jitter_40,
        _fr_40,
        color="tomato",
        alpha=0.8,
        s=60,
        label="40% MVC",
    )
    _mean_20 = np.mean(_fr_20)
    _sem_20 = np.std(_fr_20) / np.sqrt(len(_fr_20))
    plt.errorbar(
        _x_20,
        _mean_20,
        yerr=_sem_20,
        fmt="_",
        color="black",
        capsize=5,
        markersize=14,
        elinewidth=1.5,
    )
    _mean_40 = np.mean(_fr_40)
    _sem_40 = np.std(_fr_40) / np.sqrt(len(_fr_40))
    plt.errorbar(
        _x_40,
        _mean_40,
        yerr=_sem_40,
        fmt="_",
        color="black",
        capsize=5,
        markersize=14,
        elinewidth=1.5,
    )
    plt.xticks([_x_20, _x_40], ["20% MVC - Controle", "40% MVC - Controle"])
    plt.ylabel("Taxa de descarga (pps)")
    plt.title("Taxa de descarga na fase de estado estacionário")
    plt.xlim(-0.3, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=2400/Ca_cell_spike_ref_220_Umax=2400.csv",
        delimiter=",",
    )
    _data["spike_time"] = _data["spike_time"].str.replace(" ms", "").astype("float")
    _data = _data.values
    _data1 = pd.read_csv(
        f"Notebooks/data/umax_experiments/spikedata2.Umax=2400/Ca_cell_spike_ref_440_Umax=2400.csv",
        delimiter=",",
    )
    _data1["spike_time"] = _data1["spike_time"].str.replace(" ms", "").astype("float")
    _data1 = _data1.values
    _t_start_20, _t_end_20 = (4500, 54500)
    _window_20 = (_t_end_20 - _t_start_20) / 1000
    _steady_data_20 = _data[(_data[:, 1] >= _t_start_20) & (_data[:, 1] <= _t_end_20)]
    _t_start_40, _t_end_40 = (4500, 54500)
    _window_40 = (_t_end_40 - _t_start_40) / 1000
    _steady_data_40 = _data1[
        (_data1[:, 1] >= _t_start_40) & (_data1[:, 1] <= _t_end_40)
    ]
    _unique_neurons = np.unique(_data[:, 0])
    unique_neurons1 = np.unique(_data1[:, 0])
    _neurons_20 = np.random.choice(_unique_neurons, size=11, replace=True)
    print(_neurons_20)
    _neurons_40 = np.random.choice(unique_neurons1, size=9, replace=True)
    print(_neurons_40)
    _fr_20 = []
    for _neuron in _neurons_20:
        _n_spikes = np.sum(_steady_data_20[:, 0] == _neuron)
        _fr = _n_spikes / _window_20
        _fr_20.append(_fr)
    _fr_40 = []
    for _neuron in _neurons_40:
        _n_spikes = np.sum(_steady_data_40[:, 0] == _neuron)
        _fr = _n_spikes / _window_40
        _fr_40.append(_fr)
    plt.figure(figsize=(6, 5))
    _x_20 = 0
    _x_40 = 0.7
    _jitter_20 = np.random.uniform(-0.05, 0.05, size=len(_fr_20))
    _jitter_40 = np.random.uniform(-0.05, 0.05, size=len(_fr_40))
    plt.scatter(
        np.full(len(_fr_20), _x_20) + _jitter_20,
        _fr_20,
        color="royalblue",
        alpha=0.8,
        s=60,
        label="20% MVC",
    )
    plt.scatter(
        np.full(len(_fr_40), _x_40) + _jitter_40,
        _fr_40,
        color="tomato",
        alpha=0.8,
        s=60,
        label="40% MVC",
    )
    _mean_20 = np.mean(_fr_20)
    _sem_20 = np.std(_fr_20) / np.sqrt(len(_fr_20))
    plt.errorbar(
        _x_20,
        _mean_20,
        yerr=_sem_20,
        fmt="_",
        color="black",
        capsize=5,
        markersize=14,
        elinewidth=1.5,
    )
    _mean_40 = np.mean(_fr_40)
    _sem_40 = np.std(_fr_40) / np.sqrt(len(_fr_40))
    plt.errorbar(
        _x_40,
        _mean_40,
        yerr=_sem_40,
        fmt="_",
        color="black",
        capsize=5,
        markersize=14,
        elinewidth=1.5,
    )
    plt.xticks([_x_20, _x_40], ["20% MVC - Controle", "40% MVC - Controle"])
    plt.ylabel("Taxa de descarga (pps)")
    plt.title("Taxa de descarga na fase de estado estacionário")
    plt.xlim(-0.3, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
