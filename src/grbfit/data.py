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
    ].copy()

    # 🧠 identify upper limits
    is_detection = (fitdata["flux"] > 0) & (fitdata["err"] > 0)
    is_upper = ~is_detection

    # ✅ detections (used for fitting)
    det = fitdata[is_detection]

    t = det["obsdate"].values
    nu = det["freq"].values
    print("Freq range (GHz):", nu.min(), nu.max())

    y = det["flux"].values * 1e-6
    yerr = np.sqrt(det["err"]**2 + det["rms"]**2).values * 1e-6

    # 📦 also return upper limits for plotting
    upper = fitdata[is_upper]

    return (t, nu), y, yerr, upper
