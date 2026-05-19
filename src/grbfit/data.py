import pandas as pd
import numpy as np

SECONDS_PER_DAY = 24 * 60 * 60
KEV_TO_HZ = 2.417989242e17
JY_IN_CGS = 1e-23
XRT_NU_LOWER_HZ = 0.3 * KEV_TO_HZ
XRT_NU_UPPER_HZ = 10.0 * KEV_TO_HZ
XRT_CENTER_FREQ_GHZ = np.sqrt(XRT_NU_LOWER_HZ * XRT_NU_UPPER_HZ) * 1e-9
SWIFT_COLUMNS = ["obsdate", "freq", "flux", "err", "rms", "instrument"]


def _xrt_band_flux_to_fnu(band_flux, beta):
    """Convert integrated 0.3-10 keV flux to F_nu at the geometric band center."""
    nu_lower = XRT_NU_LOWER_HZ
    nu_upper = XRT_NU_UPPER_HZ

    if np.isclose(beta, 1.0):
        return (
            band_flux
            / np.sqrt(nu_lower * nu_upper)
            / np.log(nu_upper / nu_lower)
        )

    if beta > 1.0:
        denominator = 1.0 - (nu_upper / nu_lower) ** (-(beta - 1.0))
        return (
            ((beta - 1.0) * band_flux / nu_lower)
            * (nu_lower / nu_upper) ** (beta / 2.0)
            / denominator
        )

    denominator = 1.0 - (nu_lower / nu_upper) ** (1.0 - beta)
    return (
        ((1.0 - beta) * band_flux / nu_upper)
        * (nu_upper / nu_lower) ** (beta / 2.0)
        / denominator
    )


def _xrt_band_flux_to_microjy(band_flux, beta, absorption_ratio):
    # absorption_ratio is unabsorbed flux / observed flux from the XRT spectrum.
    corrected_band_flux = band_flux * absorption_ratio
    fnu_cgs = _xrt_band_flux_to_fnu(corrected_band_flux, beta)
    return (fnu_cgs / JY_IN_CGS) * 1e6


def load_swift_data(filepath, xrt_photon_index, absorption_ratio):
    if xrt_photon_index is None:
        raise ValueError("xrt_photon_index is required when loading XRT data.")
    if absorption_ratio is None:
        raise ValueError("absorption_ratio is required when loading XRT data.")

    data = []
    beta = float(xrt_photon_index) - 1.0
    absorption_ratio = float(absorption_ratio)

    if absorption_ratio <= 0:
        raise ValueError("data.absorption_ratio must be greater than 0.")

    reading_pc = False

    with open(filepath, "r") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            line_upper = line.upper()

            if not reading_pc:
                if line.startswith("!") and line[1:].strip().lower() == "pc_incbad":
                    reading_pc = True
                continue

            if line.startswith("!") or line_upper.startswith("NO"):
                break

            if line_upper.startswith("READ"):
                continue

            try:
                values = list(map(float, line.split()))
                time = values[0] / SECONDS_PER_DAY
                band_flux = values[3]
                err_plus = values[4]
                err_minus = abs(values[5])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Could not parse XRT data line {line_number} in {filepath!r}; "
                    "expected at least 6 numeric columns."
                ) from exc

            flux = _xrt_band_flux_to_microjy(band_flux, beta, absorption_ratio)
            err = 0.5 * (
                _xrt_band_flux_to_microjy(abs(err_plus), beta, absorption_ratio)
                + _xrt_band_flux_to_microjy(err_minus, beta, absorption_ratio)
            )

            data.append({
                "obsdate": time,
                "freq": XRT_CENTER_FREQ_GHZ,
                "flux": flux,
                "err": err,
                "rms": 0.1 * abs(flux),
                "instrument": "XRT",
            })

    return pd.DataFrame(data, columns=SWIFT_COLUMNS)


def _require_xrt_config(cfg):
    missing = []

    if cfg["data"].get("xrt_photon_index") is None:
        missing.append("data.xrt_photon_index")

    if cfg["data"].get("absorption_ratio") is None:
        missing.append("data.absorption_ratio")

    if missing:
        raise ValueError(
            "An XRT data file was provided, so config.yaml must also define "
            f"{' and '.join(missing)}."
        )


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
        _require_xrt_config(cfg)
        try:
            swift_df = load_swift_data(
                cfg["data"]["batxrt_file"],
                cfg["data"]["xrt_photon_index"],
                cfg["data"]["absorption_ratio"],
            )
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
                (fitdata["instrument"] == "XRT")) # Include the x-ray data
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
