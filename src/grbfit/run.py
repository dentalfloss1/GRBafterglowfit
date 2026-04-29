import yaml  # 📄 config loader
from importlib import resources
import numpy as np  # 🔢 math engine
import matplotlib.pyplot as plt  # 📊 plotting
import matplotlib.cm as cm
import corner  # 🔺 posterior visualization

from grbfit.data import load_data, prepare_fit_data  # 📥 data pipeline
from grbfit.fit import run_mcmc, make_model  # 🔧 fitting machinery
import os  # 📂 file system checks
import sys  # 🚪 clean exit
import pandas as pd

def create_template_config(path="config.yaml"):
    print("📝 No config.yaml found — creating template...")

    try:
        template = resources.files("grbfit").joinpath("config_template.yaml").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        template = """# 🔧 GRB Fit Configuration File

burst:
  name: your_grb_name_here
  t0: 1.0
  t0_rev: 0.05
  fitstart: 0.01
  z: null

data:
  radio_file: grbmeas.csv
  other_file: otherdata.txt
  batxrt_file: batxrt.txt

model:
  type: forward_only   # options: forward_only, forward_reverse
  k: 2
  p: 2.2

fit:
  initial_guess:
    f0: 1e-3
    f0_rev: 5e-5
    nua0_rev: 10
    num0_rev: 100
    nuc0_rev: 8e8
    nua_0: 13
    num_0: 100
    nuc_0: 8e8
    t_j: null
  bounds:
    # Set lower and upper equal to remove a parameter from fitting.
    # The fixed value will be the corresponding initial_guess value.
    f0: [1e-6, 1]
    f0_rev: [3e-5, 1e5]
    nua0_rev: [0.1, 1e6]
    num0_rev: [0.1, 1e6]
    nuc0_rev: [1e8, 2e9]
    nua_0: [6, 15]
    num_0: [33, 1e4]
    nuc_0: [1e8, 2e9]
    t_j: [1, 1]
  max_rest_freq: 2.47e6 # GHz, FUV in rest frame, only used in z is defined.
  fit_xrt: false
"""

    with open(path, "w") as f:
        f.write(template)

    print(f"✅ Template written to {path}")
    print(f"👉 Please fill out the fit parameters in '{path}' and rerun.\n")

# 🧠 Summarize posterior distributions
def summarize_chain(samples, keys):
    print("\n=== 📊 POSTERIOR SUMMARY ===")
    for i, k in enumerate(keys):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])  # 📏 credible intervals
        q = np.diff(mcmc)
        print(f"{k} = {mcmc[1]:.4e} +{q[1]:.4e} -{q[0]:.4e}")


# 🔺 Corner plot (parameter correlations + constraints)
def plot_corner(samples, keys):
    print("📈 Generating corner plot...")
    fig = corner.corner(samples, labels=keys, show_titles=True, title_fmt=".2e")
    plt.savefig("corner.png", dpi=200)  # 💾 save plot
    plt.close()
    print("✅ corner.png saved")

def label_helper(freq):
    if freq < 1000:
        return f"{freq:.2f} GHz"
    else:
        return f"{freq*1e9:.2e} Hz"
# 🧠 Physical grouping (GHz)
def group_frequencies_physical(nu):
    groups = {
        "Radio": [],
        "mm/sub-mm": [],
        "IR/Optical": [],
        "X-ray": []
    }

    for f in np.unique(nu):
        if f < 100:
            groups["Radio"].append(f)
        elif f < 1000:
            groups["mm/sub-mm"].append(f)
        elif f < 1e6:
            groups["IR/Optical"].append(f)
        else:
            groups["X-ray"].append(f)

    # return only non-empty groups
    return [(k, np.array(v)) for k, v in groups.items() if len(v) > 0]


# 🔧 Split each physical group into small bins
def split_group(group, max_per_bin=3):
    bins = []
    for i in range(0, len(group), max_per_bin):
        bins.append(group[i:i+max_per_bin])
    return bins


# 🚀 Final bin builder
def make_frequency_bins(nu, max_per_bin=3):
    phys_groups = group_frequencies_physical(nu)

    final_bins = []
    for name, g in phys_groups:
        sub_bins = split_group(g, max_per_bin)
        for sb in sub_bins:
            final_bins.append((name, sb))

    return final_bins

MARKERS = ["o", "s", "D", "P", "X", "*"]


COLOR_CYCLE = [
    "#000000",  # black
    "#D55E00",  # strong red-orange (colorblind-safe)
    "#0072B2",  # navy blue
]

def plot_posterior_models(cfg, samples, xdata, ydata, yerr, upper_df, excluded_df, instrument, n_draws=50):
    # 🔍 extract median t_j if present
    t_j_median = None

    def positive_mask(values):
        values = np.asarray(values)
        return np.isfinite(values) & (values > 0)

    def positive_rows(df, column="flux"):
        return df[positive_mask(df[column].values)]

    def sentinel_upper_rows(df):
        return df[(df["flux"] == -1) & (df["err"] == -1)]

    def plot_upper_limit_rows(ax, rows, color):
        if len(rows) == 0:
            return []

        t_upper = rows["obsdate"].values
        y_upper = 3.0 * np.abs(rows["rms"].values)
        upper_ok = positive_mask(y_upper)
        t_upper = t_upper[upper_ok]
        y_upper = y_upper[upper_ok]

        if len(t_upper) == 0:
            return []

        ax.scatter(
            t_upper,
            y_upper,
            marker="v",
            color=color,
            s=40,
            alpha=0.8,
            label=None
        )
        return y_upper
    
    param_keys = cfg["fit"]["param_keys"]
    fixed_params = cfg["fit"].get("fixed_params", {})
    t_j_median = fixed_params.get("t_j")
    
    if "t_j" in param_keys:
        tj_index = param_keys.index("t_j")
        t_j_samples = samples[:, tj_index]
    
        t_j_median = np.median(t_j_samples)


    model = make_model(cfg)
    t, nu = xdata
    # 👇 collect ALL frequencies (fitted + excluded + upper)
    nu_all = np.concatenate([
        nu,
        upper_df["freq"].values if upper_df is not None else [],
        excluded_df["freq"].values if excluded_df is not None else []
    ])
    freq_bins = make_frequency_bins(nu_all, max_per_bin=3)
    nbins = len(freq_bins)

    fig, axes = plt.subplots(
        nbins, 1,
        figsize=(7, 2.5 * nbins),
        sharex=True
    )

    if nbins == 1:
        axes = [axes]

    t_all = np.concatenate([
        t,
        upper_df["obsdate"].values if upper_df is not None else [],
        excluded_df["obsdate"].values if excluded_df is not None else []
    ])
    
    nu_all_data = np.concatenate([
        nu,
        upper_df["freq"].values if upper_df is not None else [],
        excluded_df["freq"].values if excluded_df is not None else []
    ])
    # 🎲 choose posterior samples
    inds = np.random.randint(len(samples), size=n_draws)
    median_theta = np.median(samples,axis=0)
    for ax, (regime, freq_group) in zip(axes, freq_bins):
        freqs = np.sort(freq_group)
        for i, freq in enumerate(freqs):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            marker = MARKERS[i % len(MARKERS)]
        
            mask = (nu == freq) & (instrument != "XRT") & (instrument != "BAT")
        
            t_data = np.array(t[mask])
            y_data = np.array(ydata[mask])
            y_err = np.array(yerr[mask])
        
            order = np.argsort(t_data)
        
            t_data = t_data[order]
            y_data = y_data[order]
            y_err = y_err[order]

            data_ok = positive_mask(y_data) & np.isfinite(y_err) & (y_err >= 0) & (y_err < y_data)
            t_data = t_data[data_ok]
            y_data = y_data[data_ok]
            y_err = y_err[data_ok]
             
            t_min = t_all[t_all > 0].min() * 0.8
            t_max = t_all.max() * 1.2
            t_line = np.geomspace(t_min, t_max, 300)

            nu_line = np.full_like(t_line, freq)
        
            # 🍝 posterior curves
            for j in inds:
                theta = samples[j]
                y_model = model(theta, (t_line, nu_line)) * 1e6
                model_ok = positive_mask(y_model)
        
                ax.plot(
                    t_line[model_ok],
                    y_model[model_ok],
                    color=color,
                    alpha=0.08
                )


            excl_subset = pd.DataFrame()   # 👈 safe empty default

            if excluded_df is not None and len(excluded_df) > 0:
                excl_mask = (excluded_df["freq"] == freq)
                excl_subset = excluded_df[excl_mask]
            
            has_excluded = len(excl_subset) > 0
            has_detection = len(t_data) > 0
            if has_detection:
                label = label_helper(freq)
            elif has_excluded:
                label = label_helper(freq) + " (excluded)"
            else:
                label = None
            if len(t_data) > 0: 
                # 📡 data
                ax.errorbar(
                    t_data,
                    y_data * 1e6,
                    yerr=y_err * 1e6,
                    fmt=marker,
                    color=color,
                    linestyle="none",
                    markersize=5,
                    label=label
                )
            # 🚫 excluded data (not used in fit)
            if excluded_df is not None and len(excluded_df) > 0:
                excl_xrt = excl_subset[excl_subset["instrument"] == "XRT"]
                excl_bat = excl_subset[excl_subset["instrument"] == "BAT"]
                excl_other = excl_subset[
                    (excl_subset["instrument"] != "XRT") &
                    (excl_subset["instrument"] != "BAT")
                ]
                excl_other_upper = sentinel_upper_rows(excl_other)
                excl_xrt_upper = sentinel_upper_rows(excl_xrt)
                excl_bat_upper = sentinel_upper_rows(excl_bat)

                excl_other = positive_rows(excl_other)
                excl_xrt = positive_rows(excl_xrt)
                excl_bat = positive_rows(excl_bat)

                if len(excl_other) > 0:
                    t_excl = excl_other["obsdate"].values
                    y_excl = excl_other["flux"].values
                    ax.scatter(
                        t_excl,
                        y_excl,
                        marker=marker,        # 👈 visually distinct
                        color=color,
                        facecolors="none",
                        linewidths=1.5,
                        alpha=0.8,
                        s=50,
                        label=label     
                    )
                if len(excl_xrt) > 0:
                    t_excl = excl_xrt["obsdate"].values
                    y_excl = excl_xrt["flux"].values
                    ax.scatter(
                        t_excl,
                        y_excl,
                        marker="s",        # 👈 visually distinct
                        color="black",
                        facecolors="none",
                        linewidths=1.5,
                        alpha=0.8,
                        s=50,
                        label="Swift XRT (excluded)"
                    )
                if len(excl_bat) > 0:
                    t_excl = excl_bat["obsdate"].values
                    y_excl = excl_bat["flux"].values
                    ax.scatter(
                        t_excl,
                        y_excl,
                        marker="D",        # 👈 visually distinct
                        color="red",
                        facecolors="none",
                        linewidths=1.5,
                        alpha=0.8,
                        s=50,
                        label="Swift BAT (excluded)"
                    )
                plot_upper_limit_rows(ax, excl_other_upper, color)
                plot_upper_limit_rows(ax, excl_xrt_upper, "black")
                plot_upper_limit_rows(ax, excl_bat_upper, "red")

            # 📈 median model
            y_med = model(median_theta, (t_line, nu_line)) * 1e6
            med_ok = positive_mask(y_med)
            ax.plot(
                t_line[med_ok],
                y_med[med_ok],
                color=color,
                linewidth=2.5
            )
            # 📡 XRT data (instrument-based, ignore frequency grouping)
            xrt_mask = (instrument == "XRT") & (nu == freq)
            
            if np.any(xrt_mask):
                t_xrt = t[xrt_mask]
                y_xrt = ydata[xrt_mask]
                yerr_xrt = yerr[xrt_mask]
                xrt_ok = positive_mask(y_xrt) & np.isfinite(yerr_xrt) & (yerr_xrt >= 0) & (yerr_xrt < y_xrt)
                t_xrt = t_xrt[xrt_ok]
                y_xrt = y_xrt[xrt_ok]
                yerr_xrt = yerr_xrt[xrt_ok]
                if len(t_xrt) > 0:
                    ax.errorbar(
                        t_xrt,
                        y_xrt * 1e6,
                        yerr=yerr_xrt * 1e6,
                        fmt="s",
                        color="black",
                        linestyle="none",
                        markersize=6,
                        label="Swift-XRT"
                    )
            bat_mask = (instrument == "BAT") & (nu == freq)

            if np.any(bat_mask):
                t_bat = t[bat_mask]
                y_bat = ydata[bat_mask]
                yerr_bat = yerr[bat_mask]
                bat_ok = positive_mask(y_bat) & np.isfinite(yerr_bat) & (yerr_bat >= 0) & (yerr_bat < y_bat)
                t_bat = t_bat[bat_ok]
                y_bat = y_bat[bat_ok]
                yerr_bat = yerr_bat[bat_ok]
                if len(t_bat) > 0:
                    ax.errorbar(
                        t_bat,
                        y_bat * 1e6,
                        yerr=yerr_bat * 1e6,
                        fmt="D",
                        color="red",
                        linestyle="none",
                        markersize=6,
                        label="Swift-BAT"
                    )
            # 🔻 upper limits (3σ)
            if upper_df is not None and len(upper_df) > 0:
                upper_mask = upper_df["freq"] == freq
                upper_subset = upper_df[upper_mask]
            
                if len(upper_subset) > 0:
                    t_upper = upper_subset["obsdate"].values
                    y_upper = 3.0 * np.abs(upper_subset["rms"].values) * 1e-6  # convert to Jy
                    upper_ok = positive_mask(y_upper)
                    t_upper = t_upper[upper_ok]
                    y_upper = y_upper[upper_ok]
            
                    if len(t_upper) > 0:
                        ax.scatter(
                            t_upper,
                            y_upper * 1e6,   # back to μJy for plotting
                            marker="v",
                            color="gray",
                            s=40,
                            alpha=0.8,
                            label=None  # avoid cluttering legend
                        )
                
                        # optional: small vertical line to indicate limit
                        for tu, yu in zip(t_upper, y_upper * 1e6):
                            ax.plot(
                                [tu, tu],
                                [yu * 0.5, yu],
                                color="gray",
                                alpha=0.6,
                                linewidth=1
                            )
                   
           # 📍 plot jet break location
        if t_j_median is not None:
            ax.axvline(
                t_j_median,
                linestyle="--",
                color=color,
                alpha=0.5,
                linewidth=1
            )

        y_all = []

        # ✅ fitted data
        for freq in freq_group:
            mask = nu == freq
            y = ydata[mask]
            err = yerr[mask]

            plot_mask = positive_mask(y) & np.isfinite(err) & (err >= 0) & (err < y)
            y_all.extend(y[plot_mask] * 1e6)
        
        # ✅ upper limits
        if upper_df is not None and len(upper_df) > 0:
            for freq in freq_group:
                mask = upper_df["freq"] == freq
                if np.any(mask):
                    y_upper = 3.0 * np.abs(upper_df["rms"].values[mask])
                    y_all.extend(y_upper[positive_mask(y_upper)])
        
        # ✅ excluded data
        if excluded_df is not None and len(excluded_df) > 0:
            for freq in freq_group:
                mask = excluded_df["freq"] == freq
                if np.any(mask):
                    y_excl = excluded_df["flux"].values[mask]
                    excl_upper = sentinel_upper_rows(excluded_df[mask])
                    y_all.extend(y_excl[positive_mask(y_excl)])
                    y_all.extend(3.0 * np.abs(excl_upper["rms"].values))
        
        y_all = np.array(y_all)
        y_all = y_all[positive_mask(y_all)]
        if len(y_all) == 0:
            continue
        
        # base limits from data
        ymin = y_all.min() * 0.5
        ymax = y_all.max() * 2
        
        # enforce minimum 2 decades
        min_ratio = 1e2
        current_ratio = ymax / ymin
        
        if current_ratio < min_ratio:
            center = np.sqrt(ymin * ymax)  # geometric mean (log-center)
            half_span = np.sqrt(min_ratio)
            ymin = center / half_span
            ymax = center * half_span
        
        ax.set_ylim(ymin, ymax)
        # 📏 axes
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 🏷️ label by regime
        ax.set_ylabel(regime)

        # legend per panel (small)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Days")

    plt.tight_layout()
    plt.savefig("posterior_models.png", dpi=200)
    plt.close()

    print("✅ posterior_models.png saved")
def normalize_config(cfg):

    if "model" not in cfg or "type" not in cfg["model"]:
        raise ValueError("❌ config.yaml must specify model.type (forward_only or forward_reverse)")

    # 🔢 model params
    cfg["model"]["k"] = int(cfg["model"]["k"])
    cfg["model"]["p"] = float(cfg["model"]["p"])

    # 🔢 burst
    cfg["burst"]["t0"] = float(cfg["burst"]["t0"])
    
    cfg["burst"]["t0_rev"] = float(cfg["burst"]["t0_rev"])
    
    cfg["burst"]["fitstart"] = float(cfg["burst"]["fitstart"])
    
    cfg["fit"]["max_rest_freq"] = float(cfg["fit"]["max_rest_freq"])
    
    cfg["fit"]["fit_xrt"] = bool(cfg["fit"].get("fit_xrt",False))

    # 🔢 initial guesses
    for k, v in cfg["fit"]["initial_guess"].items():
        if v is not None:
            cfg["fit"]["initial_guess"][k] = float(v)

    # 🔢 bounds
    for k, (low, high) in cfg["fit"]["bounds"].items():
        cfg["fit"]["bounds"][k] = [float(low), float(high)]

    return cfg

# 🚀 Main execution pipeline
def main():
    print("🚀 Starting GRB fit pipeline...")

    # 📂 check for config file
    config_path = "config.yaml"

    if not os.path.exists(config_path):
        create_template_config(config_path)
        sys.exit(0)  # 🚪 exit cleanly

    # 📄 load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = normalize_config(cfg)
    print("📄 Config loaded")
    # 📥 load + prep data
    df = load_data(cfg)
    print(f"📊 Loaded {len(df)} data points")

    xdata, ydata, yerr, upper_df, excluded_df, instrument = prepare_fit_data(df, cfg)
    print("🧹 Data prepared for fitting")

    # 🔥 run MCMC
    print("🔥 Running MCMC sampler...")
    keys, sampler = run_mcmc(cfg, xdata, ydata, yerr)
    print("✅ MCMC complete")

    # ✂️ burn-in removal
    print("✂️ Removing burn-in + thinning chain...")
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    print(f"📦 Final sample size: {len(flat_samples)}")
    print("Chain std:", np.std(flat_samples, axis=0))

    # 🧠 summarize posterior
    summarize_chain(flat_samples, keys)

    # 🔺 corner plot
    plot_corner(flat_samples, keys)

    # 🍝 posterior predictive plot
    plot_posterior_models(cfg, flat_samples, xdata, ydata, yerr, upper_df, excluded_df, instrument)

    print("🎉 All done!")


# ▶️ entry point
if __name__ == "__main__":
    main()
