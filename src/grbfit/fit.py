import numpy as np
import emcee
from grbfit.models import forward_model, forward_reverse_model


def build_param_vector(cfg):
    model_type = cfg["model"]["type"]

    if model_type == "forward_only":
        keys = ["f0", "nua_0", "num_0", "nuc_0"]
    else:
        keys = ["f0", "f0_rev", "nua0_rev", "nua_0", "num_0", "nuc_0"]

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

    if cfg["model"]["type"] == "forward_only":

        def model(theta, ivar):
            f0, nua_0, num_0, nuc_0 = theta
            return forward_model(ivar, f0, nua_0, num_0, nuc_0, k, t0, p)

    else:

        def model(theta, ivar):
            f0, f0_rev, nua0_rev, nua_0, num_0, nuc_0 = theta
            return forward_reverse_model(
                ivar, f0, f0_rev, nua0_rev,
                nua_0, num_0, nuc_0,
                k, t0, p
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
    eps = 1e-30  # avoid log(0)

    log_data = np.log10(ydata + eps)
    log_model = np.log10(model_vals + eps)
    
    log_err = yerr / (ydata + eps)  # fractional error
    
    return -0.5 * np.sum(((log_data - log_model) / log_err) ** 2)


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

    # Initialize walkers around initial guess
    keys, p0, bounds = build_param_vector(cfg)

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
