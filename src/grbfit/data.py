import pandas as pd
import numpy as np


def load_data(cfg):
    files = []

    data_cfg = cfg.get("data", {})

    # 📡 radio (required)
    if "radio_file" in data_cfg:
        files.append(pd.read_csv(data_cfg["radio_file"]))

    # 📄 optional files
    for key in ["other_file", "batxrt_file"]:
        if key in data_cfg and data_cfg[key]:
            try:
                df = pd.read_csv(data_cfg[key])
                files.append(df)
            except FileNotFoundError:
                print(f"⚠️ Warning: {data_cfg[key]} not found, skipping")

    if len(files) == 0:
        raise ValueError("❌ No data files provided in config")

    # 📦 combine everything
    data = pd.concat(files, ignore_index=True)

    # optional: sort for sanity
    data = data.sort_values(by="freq").copy()

    return data

def prepare_fit_data(df):
# TODO: replace with config-driven filtering
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
