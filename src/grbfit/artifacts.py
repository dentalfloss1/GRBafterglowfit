import json
import math

import numpy as np

from grbfit.fit import model_parameter_keys


PARAMETER_UNITS = {
    "f0": "Jy",
    "f0_rev": "Jy",
    "nua0_rev": "GHz",
    "num0_rev": "GHz",
    "nuc0_rev": "GHz",
    "nua_0": "GHz",
    "num_0": "GHz",
    "nuc_0": "GHz",
    "t_j": "d",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def summarize_parameter(cfg, samples, key):
    param_keys = cfg["fit"].get("param_keys", [])
    fixed_params = cfg["fit"].get("fixed_params", {})

    if key in param_keys:
        sample_values = samples[:, param_keys.index(key)]
        lower, median, upper = np.percentile(sample_values, [16, 50, 84])
        return {
            "value": float(median),
            "errneg": float(median - lower),
            "errpos": float(upper - median),
            "status": "fitted",
            "unit": PARAMETER_UNITS.get(key),
        }

    if key in fixed_params:
        return {
            "value": float(fixed_params[key]),
            "errneg": None,
            "errpos": None,
            "status": "fixed",
            "unit": PARAMETER_UNITS.get(key),
        }

    if key == "p":
        return {
            "value": float(cfg["model"]["p"]),
            "errneg": None,
            "errpos": None,
            "status": "fixed",
            "unit": None,
        }

    return {
        "value": None,
        "errneg": None,
        "errpos": None,
        "status": "unused",
        "unit": PARAMETER_UNITS.get(key),
    }


def build_model_fit_artifact(cfg, samples, goodness_metrics):
    parameters = {
        key: summarize_parameter(cfg, samples, key)
        for key in model_parameter_keys(cfg)
    }
    for key, frequency in cfg["fit"].get("extinction_bands", {}).items():
        parameters[key]["unit"] = "dimensionless"
        parameters[key]["observed_frequency_GHz"] = frequency
        parameters[key]["observed_frequencies_GHz"] = cfg["fit"].get(
            "extinction_band_frequencies", {}
        ).get(key, [frequency])
    fixed_parameters = [
        key for key, info in parameters.items()
        if info["status"] == "fixed"
    ]

    return {
        "format": "grbfit.model_fit",
        "format_version": 1,
        "burst": {
            "name": cfg["burst"].get("name", ""),
            "t0": cfg["burst"]["t0"],
            "t0_rev": cfg["burst"].get("t0_rev"),
            "fitstart": cfg["burst"].get("fitstart"),
            "z": cfg["burst"].get("z"),
        },
        "model": {
            "type": cfg["model"]["type"],
            "k": cfg["model"]["k"],
            "p": parameters["p"]["value"],
            "reverse_shell": cfg["model"].get("reverse_shell", "thick"),
            "fit_g": cfg["model"].get("fit_g", False),
        },
        "fit": {
            "param_keys": cfg["fit"].get("param_keys", []),
            "fixed_params": cfg["fit"].get("fixed_params", {}),
            "parameters": parameters,
            "fixed_parameters": fixed_parameters,
            "units": {
                "flux": "Jy",
                "frequency": "GHz",
                "time": "d",
            },
        },
        "goodness": goodness_metrics,
        "config": cfg,
    }


def write_model_fit_json(cfg, samples, goodness_metrics, path="model_fit.json"):
    artifact = build_model_fit_artifact(cfg, samples, goodness_metrics)
    with open(path, "w") as f:
        json.dump(_json_safe(artifact), f, indent=2)
        f.write("\n")
    print(f"Model fit JSON saved to {path}")
    return artifact


def _validate_posterior(samples, keys, metadata):
    if not isinstance(metadata, dict) or metadata.get("format") != "grbfit.posterior":
        raise ValueError("Invalid posterior archive format")
    if type(metadata.get("format_version")) is not int or metadata["format_version"] != 1:
        raise ValueError("Unsupported posterior archive format_version")
    if (keys.ndim != 1 or keys.dtype.kind != "U" or not len(keys)
            or any(not key for key in keys) or len(set(keys)) != len(keys)):
        raise ValueError("Posterior parameter names must be unique nonempty strings")
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != len(keys):
        raise ValueError("Posterior samples must be a nonempty matrix with one column per parameter")
    if samples.dtype.kind not in "fi" or not np.all(np.isfinite(samples)):
        raise ValueError("Posterior samples must contain finite real numbers")
    config = metadata.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("fit"), dict):
        raise ValueError("Posterior metadata must include the resolved fit configuration")
    if config["fit"].get("param_keys") != keys.tolist():
        raise ValueError("Posterior column names do not match the resolved configuration")
    fixed = metadata.get("fixed_params")
    if (not isinstance(fixed, dict) or set(fixed).intersection(keys)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in fixed.values())):
        raise ValueError("Posterior fixed parameters must be finite values separate from sampled columns")
    units = metadata.get("parameter_units")
    if not isinstance(units, dict) or any(
        not isinstance(units.get(key), str) or not units[key] for key in list(keys) + list(fixed)
    ):
        raise ValueError("Posterior metadata must specify units for every parameter")
    if metadata.get("sample_space") != "linear" or metadata.get("time_frame") != "observer":
        raise ValueError("Unsupported posterior sample space or time frame")
    sampler = metadata.get("sampler")
    if (not isinstance(sampler, dict) or type(sampler.get("thin")) is not int
            or sampler["thin"] < 1):
        raise ValueError("Posterior metadata must specify a positive integer thinning factor")


def write_posterior_samples(cfg, samples, sampler_diagnostics, path="posterior_samples.npz"):
    """Save retained joint draws in linear observable units, without pickle.

    Samples must already exclude burn-in and use cfg['fit']['param_keys'] order.
    Fixed parameters are stored separately, not duplicated in sample columns.
    """
    keys = np.asarray(cfg["fit"]["param_keys"], dtype=str)
    samples = np.asarray(samples)
    fixed = dict(cfg["fit"].get("fixed_params", {}))
    if "p" not in keys and "p" not in fixed:
        fixed["p"] = cfg["model"].get("p", 2.2)
    metadata = _json_safe({
        "format": "grbfit.posterior",
        "format_version": 1,
        "sample_space": "linear",
        "time_frame": "observer",
        "frequency_frame": "observer",
        "burn_in_removed": True,
        "fixed_params": fixed,
        "parameter_units": {
            key: PARAMETER_UNITS.get(key, "dimensionless")
            for key in list(keys) + list(fixed)
        },
        "config": cfg,
        "sampler": {"thin": 1, **sampler_diagnostics},
    })
    _validate_posterior(samples, keys, metadata)
    # An explicit file handle preserves the requested filename exactly.
    with open(path, "wb") as stream:
        np.savez_compressed(
            stream, samples=samples, param_keys=keys,
            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
        )
    print(f"Posterior samples saved to {path}")


def load_posterior_samples(path="posterior_samples.npz"):
    """Return (samples, ordered parameter names, metadata) from a v1 archive.

    Each row is a joint draw in linear units. This loader needs neither the
    original config.yaml nor model_fit.json and never enables pickle loading.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            samples = archive["samples"]
            keys = archive["param_keys"]
            encoded = archive["metadata_json"]
            if encoded.shape != () or encoded.dtype.kind != "U":
                raise ValueError("Posterior metadata_json must be a scalar JSON string")
            metadata = json.loads(encoded.item())
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid posterior archive: {exc}") from exc
    _validate_posterior(samples, keys, metadata)
    return samples, keys.tolist(), metadata
