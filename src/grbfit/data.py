import pandas as pd
import numpy as np


def load_data(cfg):
    plotdata = pd.read_csv(cfg["data"]["radio_file"])
    otherdata = pd.read_csv(cfg["data"]["other_file"])

    plotdata = plotdata.sort_values(by="freq").copy()

    data = pd.concat([plotdata, otherdata])

    return data


def prepare_fit_data(df):
    fitdata = df[
        (df["freq"] < 4.2e5) |
        ((df["freq"] > 1e9) & (df["freq"] < 1.5e9))
    ]

    t = fitdata["obsdate"].values
    nu = fitdata["freq"].values
    y = fitdata["flux"].values * 1e-6
    yerr = np.sqrt(fitdata["err"]**2 + fitdata["rms"]**2) * 1e-6

    return (t, nu), y, yerr
