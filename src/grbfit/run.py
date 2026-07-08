import yaml  # 📄 config loader
from importlib import resources
import numpy as np  # 🔢 math engine
import matplotlib.pyplot as plt  # 📊 plotting
import matplotlib.cm as cm
import corner  # 🔺 posterior visualization

from grbfit.data import load_data, prepare_fit_data  # 📥 data pipeline
from grbfit.fit import (
    run_mcmc,
    make_model,
    samples_to_physical,
)  # 🔧 fitting machinery
import os  # 📂 file system checks
import sys  # 🚪 clean exit
import pandas as pd
import csv
from grbfit.artifacts import write_model_fit_json

def create_template_config(path="config.yaml"):
    print("📝 No config.yaml found — creating template...")

    try:
        template = resources.files("grbfit").joinpath("config_template.yaml").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        template = """# 🔧 GRB Fit Configuration File

burst:
  name: your_grb_name_here
  # Times are in observer-frame days.
  t0: 1.0
  t0_rev: 0.05
  fitstart: 0.01
  z: null

data:
  # Input CSV flux, err, and rms columns are expected in microJy.
  # Input freq columns are expected in GHz.
  radio_file: grbmeas.csv
  other_file: otherdata.txt
  batxrt_file: null
  xrt_photon_index: null
  # Unabsorbed flux / observed flux from the XRT spectrum.
  absorption_ratio: null

model:
  type: forward_only   # options: forward_only, forward_reverse
  k: 2
  p: 2.2

fit:
  initial_guess:
    # Flux normalizations are in Jy.
    f0: 1e-3
    f0_rev: 5e-5
    # Break frequencies are in GHz.
    nua0_rev: 10
    num0_rev: 100
    nuc0_rev: 8e8
    nua_0: 13
    num_0: 100
    nuc_0: 8e8
    # Jet-break time is in observer-frame days.
    t_j: null
  bounds:
    # Set lower and upper equal to remove a parameter from fitting.
    # The fixed value will be the corresponding initial_guess value.
    # Flux normalization bounds are in Jy.
    f0: [1e-6, 1]
    f0_rev: [3e-5, 1e5]
    # Break-frequency bounds are in GHz.
    nua0_rev: [0.1, 1e6]
    num0_rev: [0.1, 1e6]
    nuc0_rev: [1e8, 2e9]
    nua_0: [6, 15]
    num_0: [33, 1e4]
    nuc_0: [1e8, 2e9]
    # Jet-break time bounds are in observer-frame days.
    t_j: [1, 1]
  max_rest_freq: 2.47e6 # GHz, FUV in rest frame, only used in z is defined.
  fit_xrt: false
  # MCMC controls. Use mcmc_mode: fixed for a quick fixed-length run.
  mcmc_mode: adaptive
  nwalkers: 32
  burn_in: 1000
  nsteps: 2000
  max_steps: 50000
  check_interval: 1000
  autocorr_ratio: 50
  autocorr_tol: 0.02
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


def calculate_goodness_of_fit(cfg, samples, xdata, ydata, yerr):
    theta = np.median(samples, axis=0)
    model = make_model(cfg)
    model_vals = model(theta, xdata)
    eps = 1e-30

    valid = (
        np.isfinite(ydata)
        & np.isfinite(yerr)
        & np.isfinite(model_vals)
        & (ydata > 0)
        & (yerr > 0)
        & (model_vals > 0)
    )

    ndata = int(np.sum(valid))
    nfit = len(cfg["fit"]["param_keys"])
    dof = ndata - nfit

    metrics = {
        "ndata": ndata,
        "nfit": nfit,
        "DOF": dof,
        "chisq": np.nan,
        "redchisq": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
    }

    if ndata == 0:
        return metrics

    log_data = np.log10(ydata[valid] + eps)
    log_model = np.log10(model_vals[valid] + eps)
    log_err = yerr[valid] / ((ydata[valid] + eps) * np.log(10))
    chi_square = np.sum(((log_data - log_model) / log_err) ** 2)

    metrics["chisq"] = chi_square
    metrics["redchisq"] = chi_square / dof if dof > 0 else np.nan
    metrics["AIC"] = chi_square + 2 * nfit
    metrics["BIC"] = chi_square + nfit * np.log(ndata)
    return metrics


def format_goodness_of_fit_report(cfg, samples, xdata, ydata, yerr, metrics=None):
    if metrics is None:
        metrics = calculate_goodness_of_fit(cfg, samples, xdata, ydata, yerr)

    lines = [
        "=== GOODNESS OF FIT ===",
        f"Data points used: {metrics['ndata']}",
        f"Fitted parameters: {metrics['nfit']}",
    ]

    lines.append(f"Degrees of freedom: {metrics['ndata']} - {metrics['nfit']} = {metrics['DOF']}")

    if metrics["ndata"] == 0:
        lines.extend([
            "Chi-square: nan",
            "Reduced chi-square: nan",
            "AIC: nan",
            "BIC: nan",
        ])
        return "\n".join(lines)

    lines.extend([
        f"Chi-square: {metrics['chisq']:.6g}",
        f"Reduced chi-square: {metrics['redchisq']:.6g}",
        f"AIC: {metrics['AIC']:.6g}",
        f"BIC: {metrics['BIC']:.6g}",
    ])
    return "\n".join(lines)


def print_goodness_of_fit_report(cfg, samples, xdata, ydata, yerr, metrics=None):
    report = format_goodness_of_fit_report(cfg, samples, xdata, ydata, yerr, metrics)
    print(f"\n{report}")
    return report


def write_fit_report(cfg, goodness_of_fit_report, path="fitreport.txt"):
    with open(path, "w") as f:
        f.write("=== CONFIGURATION ===\n")
        f.write(yaml.safe_dump(cfg, sort_keys=False))
        f.write("\n")
        f.write(goodness_of_fit_report)
        f.write("\n")
    print(f"✅ Fit report saved to {path}")


STANDARD_OUTPUT_PARAMETERS = [
    ("f0", "F0"),
    ("nua_0", "nua"),
    ("num_0", "num"),
    ("nuc_0", "nuc"),
    ("f0_rev", "F0_rev"),
    ("nua0_rev", "nua_rev"),
    ("num0_rev", "num_rev"),
    ("nuc0_rev", "nuc_rev"),
]


def summarize_output_parameter(cfg, samples, key):
    param_keys = cfg["fit"]["param_keys"]
    if key in param_keys:
        sample_values = samples[:, param_keys.index(key)]
        lower, median, upper = np.percentile(sample_values, [16, 50, 84])
        return median, median - lower, upper - median

    fixed_params = cfg["fit"].get("fixed_params", {})
    initial_guess = cfg["fit"].get("initial_guess", {})
    value = fixed_params.get(key, initial_guess.get(key, np.nan))
    return value, -1, -1


def write_standardized_fit_csv(cfg, samples, goodness_metrics, path="fit_summary.csv"):
    fieldnames = ["GRBname"]
    row = {"GRBname": cfg["burst"].get("name", "")}

    for key, label in STANDARD_OUTPUT_PARAMETERS:
        value, errneg, errpos = summarize_output_parameter(cfg, samples, key)
        fieldnames.extend([label, f"{label}errneg", f"{label}errpos"])
        row[label] = value
        row[f"{label}errneg"] = errneg
        row[f"{label}errpos"] = errpos

    t_j, t_jerrneg, t_jerrpos = summarize_output_parameter(cfg, samples, "t_j")
    fieldnames.extend(["t_j", "t_jerrneg", "t_jerrpos", "redchisq", "DOF", "AIC", "BIC"])
    row["t_j"] = t_j
    row["t_jerrneg"] = t_jerrneg
    row["t_jerrpos"] = t_jerrpos
    row["redchisq"] = goodness_metrics["redchisq"]
    row["DOF"] = goodness_metrics["DOF"]
    row["AIC"] = goodness_metrics["AIC"]
    row["BIC"] = goodness_metrics["BIC"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"✅ Standardized fit CSV saved to {path}")


# 🔺 Corner plot (parameter correlations + constraints)
CORNER_LABELS = {
    "f0": r"$F_0\ (\mu\mathrm{Jy})$",
    "f0_rev": r"$F_{0,\mathrm{rev}}\ (\mu\mathrm{Jy})$",
    "nua0_rev": r"$\nu_{a,0,\mathrm{rev}}\ (\mathrm{GHz})$",
    "num0_rev": r"$\nu_{m,0,\mathrm{rev}}\ (\mathrm{GHz})$",
    "nuc0_rev": r"$\nu_{c,0,\mathrm{rev}}\ (\mathrm{GHz})$",
    "nua_0": r"$\nu_{a,0}\ (\mathrm{GHz})$",
    "num_0": r"$\nu_{m,0}\ (\mathrm{GHz})$",
    "nuc_0": r"$\nu_{c,0}\ (\mathrm{GHz})$",
    "t_j": r"$t_j\ (\mathrm{d})$",
}

CORNER_SCALE = {
    "f0": 1e6,
    "f0_rev": 1e6,
}


def plot_corner(samples, keys):
    print("📈 Generating corner plot...")
    labels = [CORNER_LABELS.get(key, key) for key in keys]
    plot_samples = np.array(samples, copy=True)
    for i, key in enumerate(keys):
        plot_samples[:, i] *= CORNER_SCALE.get(key, 1.0)

    fig = corner.corner(
        plot_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2g",
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        max_n_ticks=3,
    )

    axes = np.asarray(fig.axes).reshape((len(keys), len(keys)))
    for i in range(len(keys)):
        q16, q50, q84 = np.percentile(plot_samples[:, i], [16, 50, 84])
        title = (
            f"${q50:.2g}_{{-{q50 - q16:.2g}}}"
            f"^{{+{q84 - q50:.2g}}}$"
        )
        axes[i, i].set_title(title, fontsize=12)

    plt.savefig("corner.png", dpi=200, bbox_inches="tight")  # 💾 save plot
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
            
            has_upper = upper_df is not None and len(upper_df[upper_df["freq"] == freq]) > 0
            has_excluded = len(excl_subset) > 0
            has_detection = len(t_data) > 0
            if has_detection:
                label = label_helper(freq)
            elif has_excluded:
                label = label_helper(freq) + " (excluded)"
            elif has_upper:
                label = label_helper(freq) + " (upper limit)"
            else:
                label = None
            model_label = label if has_upper and not has_detection and not has_excluded else None
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
                linewidth=2.5,
                label=model_label
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
                            color=color,
                            s=40,
                            alpha=0.8,
                            label=None  # avoid cluttering legend
                        )
                
                        # optional: small vertical line to indicate limit
                        for tu, yu in zip(t_upper, y_upper * 1e6):
                            ax.plot(
                                [tu, tu],
                                [yu * 0.5, yu],
                                color=color,
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


def plot_best_fit_residuals(cfg, samples, xdata, ydata, yerr, instrument):
    print("📉 Generating best-fit residual plot...")

    def positive_mask(values):
        values = np.asarray(values)
        return np.isfinite(values) & (values > 0)

    model = make_model(cfg)
    t, nu = xdata
    median_theta = np.median(samples, axis=0)
    model_vals = model(median_theta, xdata)

    valid = (
        positive_mask(t)
        & positive_mask(nu)
        & positive_mask(ydata)
        & positive_mask(yerr)
        & positive_mask(model_vals)
    )

    if not np.any(valid):
        print("⚠️ No valid fitted detections available for residuals; skipping residuals.png")
        return

    t_valid = t[valid]
    nu_valid = nu[valid]
    y_valid = ydata[valid]
    yerr_valid = yerr[valid]
    model_valid = model_vals[valid]
    instrument_valid = instrument[valid]

    log_residual = np.log10(y_valid) - np.log10(model_valid)
    log_sigma = yerr_valid / (y_valid * np.log(10))
    sigma_residual = log_residual / log_sigma

    finite = np.isfinite(sigma_residual) & np.isfinite(log_sigma) & (log_sigma > 0)
    if not np.any(finite):
        print("⚠️ Residual uncertainties are not finite; skipping residuals.png")
        return

    t_valid = t_valid[finite]
    nu_valid = nu_valid[finite]
    sigma_residual = sigma_residual[finite]
    instrument_valid = instrument_valid[finite]

    freq_bins = make_frequency_bins(nu_valid, max_per_bin=3)
    nbins = len(freq_bins)
    max_abs_residual = max(3.0, np.nanmax(np.abs(sigma_residual)) * 1.15)

    fig, axes = plt.subplots(
        nbins, 1,
        figsize=(7, 2.2 * nbins),
        sharex=True,
    )

    if nbins == 1:
        axes = [axes]

    for ax, (regime, freq_group) in zip(axes, freq_bins):
        for i, freq in enumerate(np.sort(freq_group)):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            marker = MARKERS[i % len(MARKERS)]
            mask = nu_valid == freq
            if not np.any(mask):
                continue

            t_band = t_valid[mask]
            residual_band = sigma_residual[mask]
            instrument_band = instrument_valid[mask]
            order = np.argsort(t_band)

            for inst in np.unique(instrument_band[order]):
                inst_mask = instrument_band[order] == inst
                inst_marker = marker
                inst_color = color
                label = label_helper(freq)

                if inst == "XRT":
                    inst_marker = "s"
                    inst_color = "black"
                    label = "Swift-XRT"
                elif inst == "BAT":
                    inst_marker = "D"
                    inst_color = "red"
                    label = "Swift-BAT"

                ax.scatter(
                    t_band[order][inst_mask],
                    residual_band[order][inst_mask],
                    marker=inst_marker,
                    color=inst_color,
                    s=36,
                    alpha=0.9,
                    label=label,
                )
        ax.axhline(0, color="0.25", linewidth=1.2)
        ax.axhline(1, color="0.65", linewidth=0.8, linestyle="--")
        ax.axhline(-1, color="0.65", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_ylabel(f"{regime}\nresidual ($\\sigma$)")
        ax.legend(fontsize=8)
        ax.set_ylim(-max_abs_residual, max_abs_residual)

    axes[-1].set_xlabel("Days")
    plt.tight_layout()
    plt.savefig("residuals.png", dpi=200)
    plt.close()

    print("✅ residuals.png saved")


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

    for key in ("xrt_photon_index", "absorption_ratio"):
        if key in cfg["data"] and cfg["data"][key] is not None:
            cfg["data"][key] = float(cfg["data"][key])

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
    sampler_diagnostics = getattr(sampler, "grbfit_diagnostics", {})
    thin = int(sampler_diagnostics.get("thin", 1))
    print(f"✂️ Flattening production chain with thin={thin}...")
    flat_samples_sampling = sampler.get_chain(discard=0, thin=thin, flat=True)
    flat_samples = samples_to_physical(flat_samples_sampling, keys)
    print(f"📦 Final sample size: {len(flat_samples)}")
    if sampler_diagnostics:
        print(
            "Sampler summary: "
            f"steps={sampler_diagnostics.get('production_steps')}, "
            f"acceptance={sampler_diagnostics.get('acceptance_fraction', np.nan):.3f}, "
            f"tau_max={sampler_diagnostics.get('tau_max', np.nan):.1f}, "
            f"converged={sampler_diagnostics.get('converged')}"
        )
    print("Chain std in physical units:", np.std(flat_samples, axis=0))

    # 🧠 summarize posterior
    summarize_chain(flat_samples, keys)
    goodness_metrics = calculate_goodness_of_fit(cfg, flat_samples, xdata, ydata, yerr)
    goodness_of_fit_report = print_goodness_of_fit_report(
        cfg,
        flat_samples,
        xdata,
        ydata,
        yerr,
        goodness_metrics,
    )

    # 🔺 corner plot
    plot_corner(flat_samples, keys)

    # 🍝 posterior predictive plot
    plot_posterior_models(cfg, flat_samples, xdata, ydata, yerr, upper_df, excluded_df, instrument)

    # 📉 best-fit residuals
    plot_best_fit_residuals(cfg, flat_samples, xdata, ydata, yerr, instrument)

    write_fit_report(cfg, goodness_of_fit_report)
    write_standardized_fit_csv(cfg, flat_samples, goodness_metrics)
    write_model_fit_json(cfg, flat_samples, goodness_metrics)

    print("🎉 All done!")


# ▶️ entry point
if __name__ == "__main__":
    main()
