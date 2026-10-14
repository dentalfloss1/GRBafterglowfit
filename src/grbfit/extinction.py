"""Configuration and observed-flux corrections for individually fitted bands."""
import numpy as np


BAND_FREQUENCY_TOLERANCE = 0.01


PARAMETER_PREFIX = "extinction_correction_"


def validate_extinction_config(cfg):
    fit = cfg["fit"]
    enabled = fit.get("fit_above_max_rest_freq", False)
    if not isinstance(enabled, bool):
        raise ValueError("fit.fit_above_max_rest_freq must be true or false.")
    if not enabled:
        return False
    try:
        z = float(cfg["burst"]["z"])
        cutoff = float(fit["max_rest_freq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fit.fit_above_max_rest_freq requires a defined redshift and finite positive max_rest_freq.") from exc
    if not np.isfinite(z) or z <= -1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("fit.fit_above_max_rest_freq requires a finite redshift greater than -1 and finite positive max_rest_freq.")
    try:
        low, high = map(float, fit.get("extinction_correction_bounds", [1, 1000]))
        initial = float(fit.get("extinction_correction_initial", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid extinction correction bounds or initial value.") from exc
    if not (np.all(np.isfinite([low, high, initial])) and 1 <= low < initial < high):
        raise ValueError("Extinction correction requires finite bounds with 1 <= lower < initial < upper.")
    fit["extinction_correction_bounds"] = [low, high]
    fit["extinction_correction_initial"] = initial
    return True


def group_band_frequencies(frequencies):
    """Group sorted frequencies with at most a 1% span relative to the lowest.

    Compare to the group minimum, not the preceding frequency, so a chain of
    nearby frequencies cannot join bands whose endpoints exceed the tolerance.
    """
    groups = []
    for frequency in np.unique(frequencies):
        frequency = float(frequency)
        if not groups or frequency > groups[-1][0] * (1 + BAND_FREQUENCY_TOLERANCE):
            groups.append([frequency])
        else:
            groups[-1].append(frequency)
    return groups


def correction_factors(cfg, params, frequencies):
    frequencies = np.asarray(frequencies)
    factors = np.ones_like(frequencies, dtype=float)
    members = cfg["fit"].get("extinction_band_frequencies", {})
    for key, frequency in cfg["fit"].get("extinction_bands", {}).items():
        # Older saved fits contain only a single exact frequency per parameter.
        mask = np.isin(frequencies, members.get(key, [frequency]))
        factors = np.where(mask, params[key], factors)
    return factors
