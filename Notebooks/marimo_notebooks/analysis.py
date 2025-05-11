import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    sys.path.append("./../")
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from src.functions import plot_disparos_neuronios

    return np, pd, plot_disparos_neuronios, plt


@app.cell
def _(np, plt):
    force_levels = np.array([55, 110, 220, 330, 440, 550, 660, 770, 880, 990, 1100])
    cv_force = np.zeros(len(force_levels))
    cv_force_red = np.zeros(len(force_levels))
    cv_force_aug = np.zeros(len(force_levels))
    sd_force = np.zeros(len(force_levels))
    sd_force_red = np.zeros(len(force_levels))
    sd_force_aug = np.zeros(len(force_levels))
    _i = 0
    for _force_level in force_levels:
        _data = np.loadtxt(
            f"Notebooks/data/umax_experiments/força2.Umax=2000/Ca_forca_ref{_force_level}_Umax=2000.csv",
            skiprows=1,
            delimiter=",",
        )
        time = _data[:, 0]
        force = _data[:, 1]
        data_red = np.loadtxt(
            f"Notebooks/data/umax_experiments/força2.Umax=1600/Ca_forca_ref{_force_level}_Umax=1600.csv",
            skiprows=1,
            delimiter=",",
        )
        time_red = data_red[:, 0]
        force_red = data_red[:, 1]
        data_aug = np.loadtxt(
            f"Notebooks/data/umax_experiments/força2.Umax=2400/Ca_forca_ref{_force_level}_Umax=2400.csv",
            skiprows=1,
            delimiter=",",
        )
        time_aug = data_aug[:, 0]
        force_aug = data_aug[:, 1]
        sd_force_aug[_i] = force_aug[time > 5000].std()
        sd_force[_i] = force[time > 5000].std()
        sd_force_red[_i] = force_red[time > 5000].std()
        cv_force_aug[_i] = force_aug[time > 5000].std() / force_aug[time > 5000].mean()
        cv_force[_i] = force[time > 5000].std() / force[time > 5000].mean()
        cv_force_red[_i] = force_red[time > 5000].std() / force_red[time > 5000].mean()
        _i = _i + 1
    plt.figure()
    plt.plot(np.log10(force_levels / 11), np.log10(sd_force), "r")
    plt.plot(np.log10(force_levels / 11), np.log10(sd_force_aug), "b")
    plt.plot(np.log10(force_levels / 11), np.log10(sd_force_red), "g")
    plt.show()
    plt.figure()
    plt.plot(force_levels / 11, cv_force, "r")
    plt.plot(force_levels / 11, cv_force_aug, "b")
    plt.plot(force_levels / 11, cv_force_red, "g")
    plt.show()
    return (force_levels,)


@app.cell
def _(pd, plt):
    _data = pd.read_csv(
        "Notebooks/data/umax_experiments/spikedata2.Umax=2000/Ca_cell_spike_ref_55_Umax=2000.csv",
        delimiter=",",
    )
    _data["spike_time"] = _data["spike_time"].str.replace(" ms", "")
    _data["spike_time"] = _data["spike_time"].astype("float")
    _data = _data.values
    plt.plot(_data[:, 1], _data[:, 0], ".")
    _data
    return


@app.cell
def _(plot_disparos_neuronios):
    import marimo as mo

    @mo.cache
    def compute_ar(spike_times):
        fr, t = plot_disparos_neuronios(
            spike_times,
            0,
            delta_t=0.00005,
            filtro_ordem=4,
            freq_corte=0.0005,
            tempo_max=30,
        )
        ar = len(fr[(t > 5) & (fr > 10)]) / len(fr[t > 5])
        return ar

    return (compute_ar,)


@app.cell
def _(compute_ar, force_levels, np, pd):
    rec_perc = np.zeros(len(force_levels))
    rec_perc_red = np.zeros(len(force_levels))
    rec_perc_aug = np.zeros(len(force_levels))
    j = 0
    for _force_level in force_levels:
        _data = pd.read_csv(
            f"Notebooks/data/umax_experiments/spikedata2.Umax=2000/Ca_cell_spike_ref_{_force_level}_Umax=2000.csv",
            delimiter=",",
        )
        _data["spike_time"] = _data["spike_time"].str.replace(" ms", "")
        _data["spike_time"] = _data["spike_time"].astype("float")
        _data = _data.values
        ar = np.zeros(100)
        for _i in range(100):
            try:
                ar[_i] = compute_ar(_data[_data[:, 0] == _i, 1])
            except:
                ar[_i] = 0
        _data = pd.read_csv(
            f"Notebooks/data/umax_experiments/spikedata2.Umax=2400/Ca_cell_spike_ref_{_force_level}_Umax=2400.csv",
            delimiter=",",
        )
        _data["spike_time"] = _data["spike_time"].str.replace(" ms", "")
        _data["spike_time"] = _data["spike_time"].astype("float")
        _data = _data.values
        ar_aug = np.zeros(100)
        for _i in range(100):
            try:
                ar_aug[_i] = compute_ar(_data[_data[:, 0] == _i, 1])
            except:
                ar_aug[_i] = 0
        _data = pd.read_csv(
            f"Notebooks/data/umax_experiments/spikedata2.Umax=1600/Ca_cell_spike_ref_{_force_level}_Umax=1600.csv",
            delimiter=",",
        )
        _data["spike_time"] = _data["spike_time"].str.replace(" ms", "")
        _data["spike_time"] = _data["spike_time"].astype("float")
        _data = _data.values
        ar_red = np.zeros(100)
        for _i in range(100):
            try:
                ar_red[_i] = compute_ar(_data[_data[:, 0] == _i, 1])
            except:
                ar_red[_i] = 0
        rec_perc[j] = (ar > 0.1).sum()
        rec_perc_aug[j] = (ar_aug > 0.1).sum()
        rec_perc_red[j] = (ar_red > 0.1).sum()
        j = j + 1
    return rec_perc, rec_perc_aug, rec_perc_red


@app.cell
def _(force_levels, plt, rec_perc, rec_perc_aug, rec_perc_red):
    plt.plot(force_levels / 1100 * 100, rec_perc, "r")
    plt.plot(force_levels / 1100 * 100, rec_perc_aug, "b")
    plt.plot(force_levels / 1100 * 100, rec_perc_red, "g")
    plt.grid()
    return


if __name__ == "__main__":
    app.run()
