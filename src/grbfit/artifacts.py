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
            "p": cfg["model"]["p"],
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
