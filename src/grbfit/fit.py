import numpy as np
import emcee
from grbfit.models import forward_model, forward_reverse_model


def build_param_vector(cfg):
    model_type = cfg["model"]["type"]

    if model_type == "forward_only":
        model_keys = ["f0", "nua_0", "num_0", "nuc_0"]
    else:
        model_keys = [
            "f0",
            "f0_rev",
            "nua0_rev",
            "num0_rev",
            "nuc0_rev",
            "nua_0",
            "num_0",
            "nuc_0",
        ]

    tj = cfg["fit"]["initial_guess"].get("t_j", None)
    
    if tj is not None:
        model_keys.append("t_j")

    if tj is not None and "t_j" not in cfg["fit"]["bounds"]:
        raise ValueError(
        "❌ t_j is specified but no bounds provided.\n"
        "Add to config:\n"
        "  bounds:\n"
        "    t_j: [min, max]"
        )

    keys = []
    fixed_params = {}
    for key in model_keys:
        if key not in cfg["fit"]["bounds"]:
            raise ValueError(f"❌ Missing bounds for fit parameter '{key}'")

        low = float(cfg["fit"]["bounds"][key][0])
        high = float(cfg["fit"]["bounds"][key][1])
        if high == low:
            fixed_params[key] = float(cfg["fit"]["initial_guess"][key])
        else:
            keys.append(key)

    cfg["fit"]["param_keys"] = keys
    cfg["fit"]["fixed_params"] = fixed_params

    # 🔢 build p0
    p0 = np.array([
        float(cfg["fit"]["initial_guess"][k])
        for k in keys
    ])

    bounds = np.array([
        [
            float(cfg["fit"]["bounds"][k][0]),
            float(cfg["fit"]["bounds"][k][1]),
        ]
        for k in keys
    ])

    return keys, p0, bounds

def make_model(cfg):
    k = cfg["model"]["k"]
    p = cfg["model"]["p"]
    t0 = cfg["burst"]["t0"]
    t0_rev = cfg["burst"]["t0_rev"]

    if cfg["model"]["type"] == "forward_only":
        def model(theta, ivar):
            params = {
                **cfg["fit"].get("fixed_params", {}),
                **dict(zip(cfg["fit"]["param_keys"], theta)),
            }
            return forward_model(
                                 ivar,
                                 params["f0"],
                                 params["nua_0"],
                                 params["num_0"],
                                 params["nuc_0"],
                                 cfg["model"]["k"],
                                 cfg["burst"]["t0"],
                                 cfg["model"]["p"],
                                 t_j=params.get("t_j", None),
                             )

    else:
        def model(theta, ivar):
            params = {
                **cfg["fit"].get("fixed_params", {}),
                **dict(zip(cfg["fit"]["param_keys"], theta)),
            }
        
            return forward_reverse_model(
                ivar,
                params["f0"],
                params["f0_rev"],
                params["nua0_rev"],
                params["num0_rev"],
                params["nuc0_rev"],
                params["nua_0"],
                params["num_0"],
                params["nuc_0"],
                cfg["model"]["k"],
                cfg["burst"]["t0"],
                cfg["burst"]["t0_rev"],
                cfg["model"]["p"],
                t_j=params.get("t_j", None),
            )
    return model


# ---------- LOG PROBABILITY ----------

def log_prior(theta, bounds):
    for val, (low, high) in zip(theta, bounds):
        if not (low < val < high):
            return -np.inf
    return 0.0


def log_likelihood(theta, model, xdata, ydata, yerr):
    model_vals = model(theta, xdata)
    # 🚨 guard against bad model output
    if not np.all(np.isfinite(model_vals)):
        return -np.inf
    
    if np.any(model_vals <= 0):
        return -np.inf
    eps = 1e-30  # avoid log(0)

    log_data = np.log10(ydata + eps)
    log_model = np.log10(model_vals + eps)
    
    log_err = yerr / (ydata + eps)  # fractional error
    log_like = -0.5 * np.sum(((log_data - log_model) / log_err) ** 2)

    if not np.isfinite(log_like):
        return -np.inf
    return log_like 


def log_probability(theta, model, xdata, ydata, yerr, bounds):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, xdata, ydata, yerr)


# ---------- MAIN SAMPLER ----------

def run_mcmc(cfg, xdata, ydata, yerr, nwalkers=32, nsteps=2000):
    keys, p0, bounds = build_param_vector(cfg)
    model = make_model(cfg)

    ndim = len(p0)
    if ndim == 0:
        raise ValueError("❌ All fit parameters are fixed; at least one parameter must have non-equal bounds for MCMC.")

    pos = []
    for _ in range(nwalkers):
        trial = []
        for val, (low, high) in zip(p0, bounds):
            spread = 0.1 * val if val != 0 else 1e-6
            trial.append(np.clip(val + spread * np.random.randn(), low, high))
        pos.append(trial)
    
    pos = np.array(pos)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(model, xdata, ydata, yerr, bounds),
    )

    sampler.run_mcmc(pos, nsteps, progress=True)

    return keys, sampler
