import argparse
import json
import os
import sys
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config_dir = os.path.expanduser("~/.config/matplotlib")
    if not os.access(matplotlib_config_dir, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="grbfit-matplotlib-")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grbfit.fit import evaluate_model_components, parameter_dict_from_theta
from grbfit.models import forward_shock_break_frequencies, reverse_shock_break_frequencies


SENSITIVITY_PRESETS = {
    0.8: {
        "threshold1": 7.11,
        "threshold1_label": r"3$\sigma$ SKA-Mid Band 1",
        "threshold2": 25.8,
        "threshold2_label": r"3$\sigma$ MeerKAT",
    },
    1.3: {
        "threshold1": 3.45,
        "threshold1_label": r"3$\sigma$ SKA-Mid Band 2",
        "threshold2": 19.8,
        "threshold2_label": r"3$\sigma$ MeerKAT",
    },
    6.55: {
        "threshold1": 2.112,
        "threshold1_label": r"3$\sigma$ SKA-Mid Band 5a",
        "threshold2": 69.0,
        "threshold2_label": r"3$\sigma$ ATCA",
    },
}


def _format_freq(freq):
    return f"{freq:g}"


def _preset_for_frequency(freq):
    for preset_freq, preset in SENSITIVITY_PRESETS.items():
        if np.isclose(freq, preset_freq, rtol=0, atol=1e-6):
            return preset.copy()
    return None


def _load_fit_artifact(path):
    with open(path) as f:
        artifact = json.load(f)

    if artifact.get("format") != "grbfit.model_fit":
        raise ValueError(
            f"{path!r} is not a grbfit model_fit JSON file. "
            "Run grbfit again to create model_fit.json."
        )
    return artifact


def _config_and_theta_from_artifact(artifact):
    cfg = artifact["config"]
    parameters = artifact["fit"]["parameters"]
    param_keys = list(parameters.keys())

    cfg["fit"]["param_keys"] = param_keys
    cfg["fit"]["fixed_params"] = {}
    theta = np.array([parameters[key]["value"] for key in param_keys], dtype=float)
    return cfg, theta


FIXED_PARAMETER_LABELS = {
    "f0": r"$F_0$",
    "f0_rev": r"$F_{0,\mathrm{rev}}$",
    "nua_0": r"$\nu_{a,0}$",
    "num_0": r"$\nu_{m,0}$",
    "nuc_0": r"$\nu_{c,0}$",
    "nua0_rev": r"$\nu_{a,0,\mathrm{rev}}$",
    "num0_rev": r"$\nu_{m,0,\mathrm{rev}}$",
    "nuc0_rev": r"$\nu_{c,0,\mathrm{rev}}$",
    "t_j": r"$t_j$",
}


def _format_fixed_parameter(key, info):
    label = FIXED_PARAMETER_LABELS.get(key, key)
    unit = info.get("unit")
    value = info.get("value")
    if unit:
        return f"{label} = {value:g} {unit}"
    return f"{label} = {value:g}"


def _component_is_relevant(component, total, fraction=0.01):
    component = np.abs(np.asarray(component))
    total = np.abs(np.asarray(total))
    finite = np.isfinite(component) & np.isfinite(total)
    if not np.any(finite):
        return False
    threshold = np.maximum(total[finite] * fraction, 1e-300)
    return bool(np.any(component[finite] > threshold))


def _frequency_depends_on_break(freq, break_values, factor=3.0):
    break_values = np.asarray(break_values)
    finite = np.isfinite(break_values) & (break_values > 0)
    if not np.any(finite):
        return False
    # Above a spectral break, the plotted flux depends on that break location.
    # Include a small buffer below the break because the model uses smooth breaks.
    return bool(np.any(freq >= break_values[finite] / factor))


def _fixed_parameter_affects_plot(key, cfg, theta, times, freq, components):
    params = parameter_dict_from_theta(cfg, theta)

    if key == "t_j":
        t_j = params.get("t_j")
        return t_j is not None and np.nanmax(times) >= t_j

    if key == "f0":
        return _component_is_relevant(components["forward"], components["total"])
    if key == "f0_rev":
        return _component_is_relevant(components["reverse"], components["total"])

    forward_break_keys = {
        "nua_0": 0,
        "num_0": 1,
        "nuc_0": 2,
    }
    if key in forward_break_keys:
        if not _component_is_relevant(components["forward"], components["total"]):
            return False
        freqs = np.full_like(times, freq, dtype=float)
        breaks = forward_shock_break_frequencies(
            (times, freqs),
            params["nua_0"],
            params["num_0"],
            params["nuc_0"],
            cfg["model"]["k"],
            cfg["burst"]["t0"],
            p=cfg["model"]["p"],
        )
        return _frequency_depends_on_break(freq, breaks[forward_break_keys[key]])

    reverse_break_keys = {
        "nua0_rev": 0,
        "num0_rev": 1,
        "nuc0_rev": 2,
    }
    if key in reverse_break_keys:
        if cfg["model"]["type"] == "forward_only":
            return False
        if not _component_is_relevant(components["reverse"], components["total"]):
            return False
        freqs = np.full_like(times, freq, dtype=float)
        breaks = reverse_shock_break_frequencies(
            (times, freqs),
            params["nua0_rev"],
            params["num0_rev"],
            params["nuc0_rev"],
            cfg["model"]["k"],
            cfg["burst"]["t0_rev"],
            p=cfg["model"]["p"],
        )
        return _frequency_depends_on_break(freq, breaks[reverse_break_keys[key]])

    return True


def fixed_parameter_warnings(
    artifact,
    cfg=None,
    theta=None,
    times=None,
    freq=None,
    components=None,
):
    warnings = []
    for key, info in artifact["fit"]["parameters"].items():
        if info.get("status") != "fixed":
            continue
        if all(item is not None for item in (cfg, theta, times, freq, components)):
            if not _fixed_parameter_affects_plot(key, cfg, theta, times, freq, components):
                continue
        warnings.append(_format_fixed_parameter(key, info))
    return warnings


def detectable_duration_days(times, flux_microjy, threshold_microjy):
    times = np.asarray(times)
    flux_microjy = np.asarray(flux_microjy)
    mid_flux = 0.5 * (flux_microjy[:-1] + flux_microjy[1:])
    durations = np.diff(times)
    return float(np.sum(durations[mid_flux > threshold_microjy]))


def _format_detectable_duration(duration_days, is_lower_bound, decimals=0):
    if decimals == 0:
        duration = str(int(round(duration_days)))
    else:
        duration = f"{duration_days:.{decimals}f}"
    return f">{duration}" if is_lower_bound else duration


def _compact_warning_text(warnings, max_items=3):
    if len(warnings) <= max_items:
        return "Fixed, not fitted:\n" + "\n".join(warnings)
    shown = "\n".join(warnings[:max_items])
    return f"Fixed, not fitted:\n{shown}\n... plus {len(warnings) - max_items} more"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a single-frequency GRB afterglow detectability plot."
    )
    parser.add_argument("--fit-json", default="model_fit.json")
    parser.add_argument("--freq", type=float, default=0.8, help="Frequency in GHz.")
    parser.add_argument("--threshold1", type=float, default=None, help="First threshold in microJy.")
    parser.add_argument("--threshold1-label", default=None)
    parser.add_argument("--threshold2", type=float, default=None, help="Second threshold in microJy.")
    parser.add_argument("--threshold2-label", default=None)
    parser.add_argument("--tmin", type=float, default=0.003, help="Minimum time in days.")
    parser.add_argument("--tmax", type=float, default=1000.0, help="Maximum time in days.")
    parser.add_argument("--npoints", type=int, default=100000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--no-warn",
        action="store_true",
        help="Do not show fixed-parameter warnings in the console or on the plot.",
    )
    return parser


def _resolve_thresholds(args):
    preset = _preset_for_frequency(args.freq) or {}
    threshold1 = args.threshold1 if args.threshold1 is not None else preset.get("threshold1")
    threshold2 = args.threshold2 if args.threshold2 is not None else preset.get("threshold2")
    threshold1_label = args.threshold1_label or preset.get("threshold1_label")
    threshold2_label = args.threshold2_label or preset.get("threshold2_label")

    missing = []
    if threshold1 is None:
        missing.append("--threshold1")
    if threshold2 is None:
        missing.append("--threshold2")
    if threshold1_label is None:
        missing.append("--threshold1-label")
    if threshold2_label is None:
        missing.append("--threshold2-label")
    if missing:
        raise ValueError(
            f"No built-in sensitivity preset exists for {args.freq:g} GHz; "
            f"provide {', '.join(missing)}."
        )

    return threshold1, threshold1_label, threshold2, threshold2_label


def make_detectability_plot(
    artifact,
    freq,
    threshold1,
    threshold1_label,
    threshold2,
    threshold2_label,
    tmin,
    tmax,
    npoints,
    output,
    no_warn=False,
):
    if tmin <= 0 or tmax <= tmin:
        raise ValueError("--tmin must be > 0 and --tmax must be greater than --tmin.")
    if threshold1 <= 0 or threshold2 <= 0:
        raise ValueError("Sensitivity thresholds must be positive.")

    cfg, theta = _config_and_theta_from_artifact(artifact)
    times = np.geomspace(tmin, tmax, num=npoints)
    freqs = np.full_like(times, freq, dtype=float)
    components = evaluate_model_components(cfg, theta, (times, freqs))

    total_microjy = components["total"] * 1e6
    forward_microjy = components["forward"] * 1e6
    reverse_microjy = components["reverse"] * 1e6

    det1days = detectable_duration_days(times, total_microjy, threshold1)
    det2days = detectable_duration_days(times, total_microjy, threshold2)
    det1_lower_bound = bool(total_microjy[-1] > threshold1)
    det2_lower_bound = bool(total_microjy[-1] > threshold2)

    positive_values = np.concatenate([
        total_microjy[total_microjy > 0],
        np.array([threshold1, threshold2], dtype=float),
    ])
    ymin = max(1e-3, np.nanmin(positive_values) / 3)
    ymax = np.nanmax(positive_values) * 3

    def component_for_plot(values):
        values = np.asarray(values)
        return np.where((values > 0) & (values >= ymin), values, np.nan)

    fig, ax = plt.subplots()
    ax.axhline(threshold1, linestyle=":", label=threshold1_label, color="green")
    ax.axhline(threshold2, linestyle=":", label=threshold2_label, color="black")
    ax.plot(times, total_microjy, alpha=1, color="black", linestyle="-", label="Model")
    ax.plot(
        times,
        component_for_plot(reverse_microjy),
        alpha=0.5,
        color="red",
        linestyle="--",
        label="Reverse Component",
    )
    ax.plot(
        times,
        component_for_plot(forward_microjy),
        alpha=0.5,
        color="red",
        linestyle="-.",
        label="Forward Component",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Days post-trigger")
    ax.set_ylabel(r"$\mu Jy/BM$")
    ax.grid(True)

    threshold1_name = threshold1_label.replace(r"3$\sigma$ ", "")
    threshold2_name = threshold2_label.replace(r"3$\sigma$ ", "")
    det1_title = _format_detectable_duration(det1days, det1_lower_bound)
    det2_title = _format_detectable_duration(det2days, det2_lower_bound)
    ax.set_title(
        f"{freq:g} GHz Model\n"
        f"{det1_title} days detectable with {threshold1_name}\n"
        f"{det2_title} days detectable with {threshold2_name}"
    )

    warnings = fixed_parameter_warnings(
        artifact,
        cfg=cfg,
        theta=theta,
        times=times,
        freq=freq,
        components=components,
    )
    if warnings and not no_warn:
        print("WARNING: model uses fixed parameters that were not fitted:")
        for warning in warnings:
            print(f"  - {warning}")
        ax.text(
            0.03,
            0.03,
            _compact_warning_text(warnings),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "red", "alpha": 0.85},
        )

    ax.legend()
    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    det1_console = _format_detectable_duration(det1days, det1_lower_bound, decimals=1)
    det2_console = _format_detectable_duration(det2days, det2_lower_bound, decimals=1)
    print(f"Detectable duration above {threshold1_label}: {det1_console} days")
    print(f"Detectable duration above {threshold2_label}: {det2_console} days")
    print(f"Detectability plot saved to {output}")

    return {
        "output": output,
        "det1days": det1days,
        "det2days": det2days,
        "det1_lower_bound": det1_lower_bound,
        "det2_lower_bound": det2_lower_bound,
        "fixed_parameter_warnings": warnings,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        threshold1, threshold1_label, threshold2, threshold2_label = _resolve_thresholds(args)
        artifact = _load_fit_artifact(args.fit_json)
        output = args.output or f"{_format_freq(args.freq)}GHzmodellc_meas.png"
        make_detectability_plot(
            artifact,
            args.freq,
            threshold1,
            threshold1_label,
            threshold2,
            threshold2_label,
            args.tmin,
            args.tmax,
            args.npoints,
            output,
            no_warn=args.no_warn,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
