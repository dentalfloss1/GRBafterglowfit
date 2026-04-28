import yaml  # 📄 config loader
import numpy as np  # 🔢 math engine
import matplotlib.pyplot as plt  # 📊 plotting
import matplotlib.cm as cm
import corner  # 🔺 posterior visualization

from grbfit.data import load_data, prepare_fit_data  # 📥 data pipeline
from grbfit.fit import run_mcmc, make_model  # 🔧 fitting machinery
import os  # 📂 file system checks
import sys  # 🚪 clean exit

def create_template_config(path="config.yaml"):
    print("📝 No config.yaml found — creating template...")

    template = """# 🔧 GRB Fit Configuration File

burst:
  name: your_grb_name_here
  t0: 1.0

data:
  radio_file: grbmeas.csv
  other_file: otherdata.txt
  batxrt_file: batxrt.txt

model:
  type: forward_reverse   # options: forward_only, forward_reverse
  k: 2
  p: 2.2

fit:
  initial_guess:
    f0: 1e-3
    f0_rev: 5e-5
    nua0_rev: 10
    nua_0: 13
    num_0: 100
    nuc_0: 8e8

  bounds:
    f0: [1e-6, 1]
    f0_rev: [3e-5, 1e5]
    nua0_rev: [0.1, 1e6]
    nua_0: [6, 15]
    num_0: [33, 1e4]
    nuc_0: [1e8, 2e9]
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
    fig = corner.corner(samples, labels=keys, show_titles=True)
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

def plot_posterior_models(cfg, samples, xdata, ydata, yerr, upper_df, n_draws=50):
    model = make_model(cfg)
    t, nu = xdata


    freq_bins = make_frequency_bins(nu, max_per_bin=3)
    nbins = len(freq_bins)

    fig, axes = plt.subplots(
        nbins, 1,
        figsize=(7, 2.5 * nbins),
        sharex=True
    )

    if nbins == 1:
        axes = [axes]

    # 🎲 choose posterior samples
    inds = np.random.randint(len(samples), size=n_draws)
    median_theta = np.median(samples,axis=0)
    for ax, (regime, freq_group) in zip(axes, freq_bins):
        freqs = np.sort(freq_group)
        for i, freq in enumerate(freqs):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            marker = MARKERS[i % len(MARKERS)]
        
            mask = nu == freq
        
            t_data = np.array(t[mask])
            y_data = np.array(ydata[mask])
            y_err = np.array(yerr[mask])
        
            order = np.argsort(t_data)
        
            t_data = t_data[order]
            y_data = y_data[order]
            y_err = y_err[order]
        
            t_line = np.geomspace(t_data.min(), t_data.max(), 200)
            nu_line = np.full_like(t_line, freq)
        
            # 🍝 posterior curves
            for j in inds:
                theta = samples[j]
                y_model = model(theta, (t_line, nu_line)) * 1e6
        
                ax.plot(
                    t_line,
                    y_model,
                    color=color,
                    alpha=0.08
                )
        
            # 📡 data
            ax.errorbar(
                t_data,
                y_data * 1e6,
                yerr=y_err * 1e6,
                fmt=marker,
                color=color,
                linestyle="none",
                markersize=5,
                label=label_helper(freq)
            )
        
            # 📈 median model
            y_med = model(median_theta, (t_line, nu_line)) * 1e6
            ax.plot(
                t_line,
                y_med,
                color=color,
                linewidth=2.5
            )
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
    # 🔢 model params
    cfg["model"]["k"] = int(cfg["model"]["k"])
    cfg["model"]["p"] = float(cfg["model"]["p"])

    # 🔢 burst
    cfg["burst"]["t0"] = float(cfg["burst"]["t0"])

    # 🔢 initial guesses
    for k, v in cfg["fit"]["initial_guess"].items():
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

    xdata, ydata, yerr, upper_df = prepare_fit_data(df)
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
    plot_posterior_models(cfg, flat_samples, xdata, ydata, yerr, upper_df)

    print("🎉 All done!")


# ▶️ entry point
if __name__ == "__main__":
    main()
