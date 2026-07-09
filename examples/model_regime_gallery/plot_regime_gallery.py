#!/usr/bin/env python3
"""Plot FS/RS spectral-regime diagnostics against sharp power-law guides."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grbfit_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from grbfit.models import (  # noqa: E402
    forward_shock_absorption_tau,
    forward_shock_break_frequencies,
    forward_shock_flux,
    reverse_shock,
)


F0 = 1e-3
F0_REV = 8e-4
T0 = 1.0
T0_REV = 0.05


@dataclass(frozen=True)
class RegimeCase:
    key: str
    title: str
    breaks: dict[str, float]
    order: tuple[str, str, str]
    spectral_slopes: tuple[str, str, str, str]


REGIME_CASES = (
    RegimeCase(
        key="slow",
        title=r"$\nu_a < \nu_m < \nu_c$",
        breaks={"nua": 1e2, "num": 1e6, "nuc": 1e10},
        order=("nua", "num", "nuc"),
        spectral_slopes=("2", "1/3", "-(p-1)/2", "-p/2"),
    ),
    RegimeCase(
        key="fast",
        title=r"$\nu_a < \nu_c < \nu_m$",
        breaks={"nua": 1e2, "num": 1e10, "nuc": 1e6},
        order=("nua", "nuc", "num"),
        spectral_slopes=("2", "1/3", "-1/2", "-p/2"),
    ),
    RegimeCase(
        key="self_absorbed",
        title=r"$\nu_m < \nu_a < \nu_c$",
        breaks={"nua": 1e6, "num": 1e2, "nuc": 1e10},
        order=("num", "nua", "nuc"),
        spectral_slopes=("2", "5/2", "-(p-1)/2", "-p/2"),
    ),
)


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower().lstrip(".") for item in value.split(",")]
    formats = [item for item in formats if item]
    allowed = {"png", "pdf"}
    unsupported = sorted(set(formats) - allowed)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported format(s): {', '.join(unsupported)}. Use png and/or pdf."
        )
    return formats or ["png"]


def _slope_value(label: str, p: float) -> float:
    return {
        "2": 2.0,
        "1/3": 1.0 / 3.0,
        "-1/2": -0.5,
        "5/2": 2.5,
        "-(p-1)/2": -(p - 1.0) / 2.0,
        "-p/2": -p / 2.0,
    }[label]


def _format_number(value: float) -> str:
    if abs(value) < 1e-10:
        return "0"
    return f"{value:.3g}"


def _format_slope_label(label: str, p: float) -> str:
    value = _slope_value(label, p)
    return rf"$\beta={label}={_format_number(value)}$"


def _format_temporal_label(value: float) -> str:
    return rf"$\alpha={_format_number(value)}$"


def _ordered_breaks(case: RegimeCase) -> list[float]:
    return [case.breaks[name] for name in case.order]


def _break_label(name: str) -> str:
    return {"nua": r"$\nu_a$", "num": r"$\nu_m$", "nuc": r"$\nu_c$"}[name]


def _sharp_piecewise_from_first_break(
    x: np.ndarray,
    first_break_flux: float,
    breaks: list[float],
    slopes: list[float],
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    xb1, xb2, xb3 = breaks

    first = x < xb1
    second = (x >= xb1) & (x < xb2)
    third = (x >= xb2) & (x < xb3)
    fourth = x >= xb3

    y[first] = first_break_flux * (x[first] / xb1) ** slopes[0]
    y[second] = first_break_flux * (x[second] / xb1) ** slopes[1]
    y[third] = (
        first_break_flux
        * (xb2 / xb1) ** slopes[1]
        * (x[third] / xb2) ** slopes[2]
    )
    y[fourth] = (
        first_break_flux
        * (xb2 / xb1) ** slopes[1]
        * (xb3 / xb2) ** slopes[2]
        * (x[fourth] / xb3) ** slopes[3]
    )
    return y


def _first_break_flux(case: RegimeCase, fnu_max: float) -> float:
    nua = case.breaks["nua"]
    num = case.breaks["num"]
    nuc = case.breaks["nuc"]
    if case.key == "slow":
        return fnu_max * (nua / num) ** (1.0 / 3.0)
    if case.key == "fast":
        return fnu_max * (nua / nuc) ** (1.0 / 3.0)
    if case.key == "self_absorbed":
        return fnu_max * (num / nua) ** 3
    raise ValueError(f"Unsupported regime: {case.key}")


def _shock_flux(
    shock: str,
    case: RegimeCase,
    times: np.ndarray,
    freqs: np.ndarray,
    k: int,
    p: float,
) -> np.ndarray:
    if shock == "fs":
        return forward_shock_flux(
            (times, freqs),
            F0,
            case.breaks["nua"],
            case.breaks["num"],
            case.breaks["nuc"],
            k,
            T0,
            p=p,
        )
    if shock == "rs":
        return reverse_shock(
            (times, freqs),
            F0_REV,
            case.breaks["nua"],
            case.breaks["num"],
            case.breaks["nuc"],
            k,
            T0_REV,
            p=p,
        )
    raise ValueError(f"Unsupported shock: {shock}")


def _fs_absorption_factor(times: np.ndarray, freqs: np.ndarray, k: int, p: float) -> np.ndarray:
    tau = forward_shock_absorption_tau(
        (times, freqs),
        1e4,
        1e8,
        1e12,
        k,
        T0,
        p=p,
    )
    return np.exp(-tau)


def _plot_fs_absorption_factor(p: float, n_points: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(r"Forward-shock absorption applied to reverse-shock photons")
    freqs = np.geomspace(1e0, 1e13, n_points)
    times_to_plot = np.array([0.05, 0.3, 1.0, 3.0])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    low_label = r"$\tau_{\rm FS}\propto(\nu/\nu_{a,\rm FS})^{5/3}$"
    high_label = rf"$\tau_{{\rm FS}}\propto(\nu/\nu_{{a,\rm FS}})^{{-{_format_number((p + 4) / 2)}}}$"

    for row, k in enumerate((0, 2)):
        tau_ax = axes[row, 0]
        transmission_ax = axes[row, 1]

        for idx, tval in enumerate(times_to_plot):
            times = np.full_like(freqs, tval)
            tau = forward_shock_absorption_tau(
                (times, freqs),
                1e4,
                1e8,
                1e12,
                k,
                T0,
                p=p,
            )
            transmission = np.exp(-np.clip(tau, 0.0, 700.0))
            label = rf"$t={_format_number(tval)}$"
            color = colors[idx % len(colors)]
            tau_ax.loglog(freqs, tau, lw=1.8, color=color, label=label)
            transmission_ax.semilogx(freqs, transmission, lw=1.8, color=color, label=label)

        ref_breaks = forward_shock_break_frequencies(
            (np.array([T0]), np.array([1.0])),
            1e4,
            1e8,
            1e12,
            k,
            T0,
            p=p,
        )
        nua_ref, num_ref, nuc_ref = [breaks[0] for breaks in ref_breaks]
        lower_ref = min(num_ref, nuc_ref)
        for ax in (tau_ax, transmission_ax):
            ax.axvline(nua_ref, color="0.25", lw=1.0, ls=":", label=r"$\nu_{a,\rm FS}$ at $t_0$")
            ax.axvline(
                lower_ref,
                color="0.55",
                lw=1.0,
                ls="--",
                label=r"$\min(\nu_{m,\rm FS},\nu_{c,\rm FS})$ at $t_0$",
            )
            ax.set_xlim(freqs[0], freqs[-1])
            ax.grid(True, which="both", alpha=0.2)

        tau_ax.text(0.03, 0.08, low_label + "\n" + high_label, transform=tau_ax.transAxes, fontsize=8)
        tau_ax.set_title(rf"$k={k}$ optical depth")
        tau_ax.set_ylabel(r"$\tau_{\rm FS}$")
        tau_ax.set_ylim(1e-12, 1e8)
        transmission_ax.set_title(rf"$k={k}$ transmission")
        transmission_ax.set_ylabel(r"$e^{-\tau_{\rm FS}}$")
        transmission_ax.set_ylim(-0.03, 1.03)
        transmission_ax.legend(fontsize=7, loc="best")

    axes[-1, 0].set_xlabel("Frequency")
    axes[-1, 1].set_xlabel("Frequency")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def _temporal_exponents(shock: str, case: RegimeCase, k: int, p: float) -> tuple[float, list[float]]:
    if shock == "fs":
        fmax = -k / (2.0 * (4.0 - k))
        bm = -1.5
        bc = -(4.0 - 3.0 * k) / (2.0 * (4.0 - k))
        if case.key == "slow":
            ba = -3.0 * k / (5.0 * (4.0 - k))
            norm = fmax + (ba - bm) / 3.0
            return norm, [ba, bm, bc]
        if case.key == "fast":
            ba = -(10.0 + 3.0 * k) / (5.0 * (4.0 - k))
            norm = fmax + (ba - bc) / 3.0
            return norm, [ba, bc, bm]
        if case.key == "self_absorbed":
            ba = -(12.0 * p + 8.0 - 3.0 * p * k + 2.0 * k) / (
                2.0 * (4.0 - k) * (p + 4.0)
            )
            norm = fmax + 3.0 * (bm - ba)
            return norm, [bm, ba, bc]

    if shock == "rs":
        fmax = -(47.0 - 10.0 * k) / (12.0 * (4.0 - k))
        bm = -(73.0 - 14.0 * k) / (12.0 * (4.0 - k))
        bc = bm
        if case.key in ("slow", "fast"):
            ba = -(32.0 - 7.0 * k) / (15.0 * (4.0 - k))
        else:
            ba = -(p * (73.0 - 14.0 * k) + 2.0 * (67.0 - 14.0 * k)) / (
                12.0 * (4.0 - k) * (p + 4.0)
            )

        if case.key == "slow":
            norm = fmax + (ba - bm) / 3.0
            return norm, [ba, bm, bc]
        if case.key == "fast":
            norm = fmax + (ba - bc) / 3.0
            return norm, [ba, bc, bm]
        if case.key == "self_absorbed":
            norm = fmax + 3.0 * (bm - ba)
            return norm, [bm, ba, bc]

    raise ValueError(f"Unsupported shock/regime: {shock}/{case.key}")


def _temporal_slope(norm_exp: float, break_exps: list[float], slopes: list[float], segment: int) -> float:
    if segment == 0:
        return norm_exp - slopes[0] * break_exps[0]
    if segment == 1:
        return norm_exp - slopes[1] * break_exps[0]
    if segment == 2:
        return norm_exp + slopes[1] * (break_exps[1] - break_exps[0]) - slopes[2] * break_exps[1]
    if segment == 3:
        return (
            norm_exp
            + slopes[1] * (break_exps[1] - break_exps[0])
            + slopes[2] * (break_exps[2] - break_exps[1])
            - slopes[3] * break_exps[2]
        )
    raise ValueError(f"Unsupported segment: {segment}")


def _segment_frequencies(case: RegimeCase) -> list[float]:
    xb1, xb2, xb3 = _ordered_breaks(case)
    return [xb1 / 100.0, np.sqrt(xb1 * xb2), np.sqrt(xb2 * xb3), xb3 * 100.0]


def _draw_break_lines(ax: plt.Axes, case: RegimeCase) -> None:
    for name in case.order:
        value = case.breaks[name]
        ax.axvline(value, color="0.65", lw=0.8, ls=":")
        ax.text(
            value,
            0.96,
            _break_label(name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )


def _draw_spectral_labels(ax: plt.Axes, case: RegimeCase, p: float, y_level: float) -> None:
    xb1, xb2, xb3 = _ordered_breaks(case)
    xmin, xmax = ax.get_xlim()
    positions = [
        np.sqrt(xmin * xb1),
        np.sqrt(xb1 * xb2),
        np.sqrt(xb2 * xb3),
        np.sqrt(xb3 * xmax),
    ]
    for xval, label in zip(positions, case.spectral_slopes):
        if xmin < xval < xmax:
            ax.text(
                xval,
                y_level,
                _format_slope_label(label, p),
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.25",
            )


def _plot_spectra(shock: str, p: float, n_points: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    fig.suptitle("Forward-shock spectra" if shock == "fs" else "Reverse-shock spectra")

    for row, k in enumerate((0, 2)):
        for col, case in enumerate(REGIME_CASES):
            ax = axes[row, col]
            breaks = _ordered_breaks(case)
            freqs = np.geomspace(min(breaks) / 1e3, max(breaks) * 1e3, n_points)
            tref = T0 if shock == "fs" else T0_REV
            times = np.full_like(freqs, tref)
            model = _shock_flux(shock, case, times, freqs, k, p)
            slopes = [_slope_value(label, p) for label in case.spectral_slopes]
            guide = _sharp_piecewise_from_first_break(
                freqs,
                _first_break_flux(case, F0 if shock == "fs" else F0_REV),
                breaks,
                slopes,
            )

            ax.loglog(freqs, model, color="tab:blue", lw=2.0, label="smooth model")
            if shock == "rs":
                absorbed = model * _fs_absorption_factor(times, freqs, k, p)
                ax.loglog(freqs, absorbed, color="tab:red", lw=1.8, label=r"RS $\times e^{-\tau_{\rm FS}}$")
            ax.loglog(
                freqs,
                guide,
                color="0.15",
                lw=1.4,
                ls="--",
                label="sharp BPL guide",
                zorder=5,
            )

            _draw_break_lines(ax, case)
            ax.set_title(f"k={k}, {case.title}")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Flux density")
            baseline = np.concatenate([guide, model])
            baseline = baseline[np.isfinite(baseline) & (baseline > 0)]
            ymin = np.nanmin(baseline) / 3.0
            ymax = np.nanmax(baseline) * 8.0
            ax.set_ylim(ymin, ymax)
            _draw_spectral_labels(ax, case, p, ymax / 5.0)
            ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    return fig


def _plot_light_curves(shock: str, p: float, n_points: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    fig.suptitle("Forward-shock light curves" if shock == "fs" else "Reverse-shock light curves")

    for row, k in enumerate((0, 2)):
        for col, case in enumerate(REGIME_CASES):
            ax = axes[row, col]
            tref = T0 if shock == "fs" else T0_REV
            times = tref * np.geomspace(0.3, 3.0, n_points)
            norm_exp, break_exps = _temporal_exponents(shock, case, k, p)
            spectral_slopes = [_slope_value(label, p) for label in case.spectral_slopes]
            baseline = []

            for idx, freq in enumerate(_segment_frequencies(case)):
                freqs = np.full_like(times, freq)
                model = _shock_flux(shock, case, times, freqs, k, p)
                if shock == "rs":
                    absorbed = model * _fs_absorption_factor(times, freqs, k, p)
                    ax.loglog(times, absorbed, lw=1.2, ls=":", alpha=0.9, color=f"C{idx}")
                    display_model = model
                else:
                    display_model = model
                baseline.append(display_model)

                alpha = _temporal_slope(norm_exp, break_exps, spectral_slopes, idx)
                ref_flux = np.interp(tref, times, display_model)
                guide = ref_flux * (times / tref) ** alpha
                baseline.append(guide)
                label = rf"{freq:.2g}: {_format_temporal_label(alpha)}"
                ax.loglog(times, display_model, color=f"C{idx}", lw=1.8, label=label)
                ax.loglog(times, guide, color=f"C{idx}", lw=1.0, ls="--")

            if shock == "rs":
                ax.text(
                    0.04,
                    0.05,
                    r"solid: raw RS; dotted: RS $\times e^{-\tau_{\rm FS}}$; dashed: guide",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="0.35",
                )
            ax.set_title(f"k={k}, {case.title}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Flux density")
            baseline_values = np.concatenate(baseline)
            baseline_values = baseline_values[
                np.isfinite(baseline_values) & (baseline_values > 0)
            ]
            ax.set_ylim(np.nanmin(baseline_values) / 3.0, np.nanmax(baseline_values) * 3.0)
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def build_gallery(output_dir: Path, formats: list[str], p: float, n_points: int) -> list[Path]:
    outputs = []
    outputs.extend(_save_figure(_plot_spectra("fs", p, n_points), output_dir, "forward_shock_spectra", formats))
    outputs.extend(_save_figure(_plot_light_curves("fs", p, n_points), output_dir, "forward_shock_light_curves", formats))
    outputs.extend(_save_figure(_plot_fs_absorption_factor(p, n_points), output_dir, "forward_shock_absorption_factor", formats))
    outputs.extend(_save_figure(_plot_spectra("rs", p, n_points), output_dir, "reverse_shock_spectra", formats))
    outputs.extend(_save_figure(_plot_light_curves("rs", p, n_points), output_dir, "reverse_shock_light_curves", formats))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate FS/RS regime-gallery plots against sharp broken-power-law guides."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=["png"],
        help="Comma-separated output formats: png,pdf.",
    )
    parser.add_argument("--p", type=float, default=2.2, help="Electron energy index.")
    parser.add_argument("--n-points", type=int, default=800, help="Samples per plotted curve.")
    parser.add_argument("--show", dest="show", action="store_true", help="Display figures after saving.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Save only; this is the default.")
    parser.set_defaults(show=False)
    args = parser.parse_args(argv)

    outputs = build_gallery(args.output_dir, args.formats, args.p, args.n_points)
    for path in outputs:
        print(f"saved {path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
