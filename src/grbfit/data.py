import pandas as pd
import numpy as np

def load_swift_data(filepath):
    import numpy as np

    data = []

    kev_to_hz = 2.418e17
    nu_ref = 10 * kev_to_hz  # 10 keV
    nu_ref = nu_ref*1e-9
    store = False
    current_instrument = None

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("NO") or line.upper().startswith("READ"):
                continue
            if line.startswith("!"):
                header = line.strip().lower()

                if "flux" in header:
                    if "incbad" in header:
                        isbad = True
                    else:
                        isbad = False
                    store = True

                    if "bat" in header:
                        current_instrument = "BAT"
                    elif "xrt" in header:
                        current_instrument = "XRT"
                    else:
                        current_instrument = None
                else:
                    store = False
                    current_instrument = None

                continue

            if store and current_instrument:
                values = list(map(float, line.split()))

                time = values[0]/24/60/60
                flux = values[3]*1e6
                err_plus = values[4]*1e6
                err_minus = abs(values[5])*1e6

                err = 0.5 * (err_plus + err_minus)
                if (not isbad) or (time not in [ d["obsdate"] for d in data]):
                    data.append({
                        "obsdate": time,
                        "freq": nu_ref,
                        "flux": flux,
                        "err": err,
                        "rms": 0.1*flux,
                        "instrument": current_instrument
                    })
    return pd.DataFrame(data)

def load_data(cfg):

    # 📡 radio (required)
    plotdata = pd.read_csv(cfg["data"]["radio_file"])
    plotdata["instrument"] = "radio"

    dfs = [plotdata]

    # 📄 optional files
    if "other_file" in cfg["data"]:
        try:
            otherdata = pd.read_csv(cfg["data"]["other_file"])
            otherdata["instrument"] = "other"
            dfs.append(otherdata)
        except FileNotFoundError:
            print(f"⚠️ Warning: {cfg['data']['otherfile']} not found, skipping")

    # 🔥 NEW: Swift support
    if "batxrt_file" in cfg["data"] and cfg["data"]["batxrt_file"] is not None:
        try:
            swift_df = load_swift_data(cfg["data"]["batxrt_file"])
            dfs.append(swift_df)
        except FileNotFoundError:
            print(f"⚠️ Warning: {cfg['data']['batxrt_file']} not found, skipping")

    # 📦 combine everything
    data = pd.concat(dfs, ignore_index=True)

    # optional: sort for sanity
    data = data.sort_values(by="freq").copy()

    return data

def prepare_fit_data(df, cfg):
    z = cfg["burst"].get("z", None)
    
    df = df.copy()
    
    if z is not None:
        df["freq_rest"] = df["freq"] * (1 + z)
    else:
        df["freq_rest"] = df["freq"]

    fitdata = df.copy()
    # remove BAT
    fitdata = fitdata[fitdata["instrument"] != "BAT"]
    
    fit_xrt = cfg["fit"].get("fit_xrt",False)

    # remove host-extincted region (rest frame)
    if z is not None and "max_rest_freq" in cfg["fit"]:
        if not fit_xrt:
            fitdata = fitdata[
                (fitdata["freq_rest"] < cfg["fit"]["max_rest_freq"])
                & (fitdata["obsdate"] > cfg["burst"]["fitstart"])
            ]
        else:
            fitdata = fitdata[
                ((fitdata["freq_rest"] < cfg["fit"]["max_rest_freq"]) |
                (fitdata["freq"] > 1e9)) # Include the x-ray data
                & (fitdata["obsdate"] > cfg["burst"]["fitstart"])
            ]
    print("Total points:", len(df))
    print("After cuts:", len(fitdata))

    # 🧠 identify upper limits
    is_detection = (fitdata["flux"] > 0) & (fitdata["err"] > 0)
    is_upper = ~is_detection

    # ✅ detections (used for fitting)
    det = fitdata[is_detection]

    t = det["obsdate"].values
    nu = det["freq"].values
    instrument_det = det["instrument"].values   # 👈 aligned with t, nu
    print("Freq range (GHz):", nu.min(), nu.max())

    y = det["flux"].values * 1e-6
    yerr = np.sqrt(det["err"]**2 + det["rms"]**2).values * 1e-6

    # 📦 also return upper limits for plotting
    upper = fitdata[is_upper]

    # 👉 NEW: identify excluded points
    excluded = df.loc[~df.index.isin(fitdata.index)]
    return (t, nu), y, yerr, upper, excluded, instrument_det



