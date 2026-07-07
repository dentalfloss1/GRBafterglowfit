import numpy as np
import emcee
from grbfit.models import (
    forward_model,
    forward_reverse_model,
    forward_shock_absorption_tau,
    reverse_shock,
    theory_bigsbpl,
)


LOG10_PARAMETERS = {
    "f0",
    "f0_rev",
    "nua0_rev",
    "num0_rev",
    "nuc0_rev",
    "nua_0",
    "num_0",
    "nuc_0",
    "t_j",
}


def uses_log10_sampling(key):
    return key in LOG10_PARAMETERS


def _log10_checked(key, value):
    value = float(value)
    if value <= 0:
        raise ValueError(f"Log10-sampled parameter '{key}' must be positive; got {value}")
    return np.log10(value)


def to_sampling_value(key, value):
    if uses_log10_sampling(key):
        return _log10_checked(key, value)
    return float(value)


def to_physical_value(key, value):
    if uses_log10_sampling(key):
        return 10 ** float(value)
    return float(value)


def theta_to_physical(theta, keys):
    return np.array([
        to_physical_value(key, value)
        for key, value in zip(keys, theta)
    ])


def samples_to_physical(samples, keys):
    samples = np.asarray(samples)
    physical = np.array(samples, dtype=float, copy=True)
    for i, key in enumerate(keys):
        if uses_log10_sampling(key):
            physical[:, i] = 10 ** physical[:, i]
    return physical


def model_parameter_keys(cfg):
    """Return the parameter names used by the configured model."""
    if cfg["model"]["type"] == "forward_only":
        keys = ["f0", "nua_0", "num_0", "nuc_0"]
    else:
        keys = [
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
        keys.append("t_j")

    return keys


def build_param_vector(cfg):
    model_keys = model_parameter_keys(cfg)
    tj = cfg["fit"]["initial_guess"].get("t_j", None)

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
        if key not in cfg["fit"]["initial_guess"]:
            raise ValueError(f"❌ Missing initial_guess for fit parameter '{key}'")

        low = float(cfg["fit"]["bounds"][key][0])
        high = float(cfg["fit"]["bounds"][key][1])
        initial = float(cfg["fit"]["initial_guess"][key])
        if high < low:
            raise ValueError(
                f"❌ Invalid bounds for '{key}': lower bound {low:g} is greater "
                f"than upper bound {high:g}."
            )
        if high == low:
            fixed_params[key] = initial
        else:
            if not (low < initial < high):
                raise ValueError(
                    f"❌ Initial guess for '{key}' must be inside its fit bounds.\n"
                    f"   initial_guess.{key}: {initial:g}\n"
                    f"   bounds.{key}: [{low:g}, {high:g}]\n"
                    "   Move the initial guess between the bounds, or widen the bounds."
                )
            if uses_log10_sampling(key):
                _log10_checked(key, low)
                _log10_checked(key, high)
            keys.append(key)

    cfg["fit"]["param_keys"] = keys
    cfg["fit"]["fixed_params"] = fixed_params

    # 🔢 build p0
    p0 = np.array([
        to_sampling_value(k, cfg["fit"]["initial_guess"][k])
        for k in keys
    ])

    bounds = np.array([
        [
            to_sampling_value(k, cfg["fit"]["bounds"][k][0]),
            to_sampling_value(k, cfg["fit"]["bounds"][k][1]),
        ]
        for k in keys
    ])

    return keys, p0, bounds


def parameter_dict_from_theta(cfg, theta):
    return {
        **cfg["fit"].get("fixed_params", {}),
        **dict(zip(cfg["fit"]["param_keys"], theta)),
    }


def evaluate_model_components(cfg, theta, ivar):
    """Evaluate total, forward, and reverse model components in Jy."""
    params = parameter_dict_from_theta(cfg, theta)
    k = cfg["model"]["k"]
    p = cfg["model"]["p"]
    t0 = cfg["burst"]["t0"]
    t_j = params.get("t_j", None)

    forward = theory_bigsbpl(
        ivar,
        params["f0"],
        params["nua_0"],
        params["num_0"],
        params["nuc_0"],
        k,
        t0,
        jet_break=t_j,
        p=p,
    )

    if cfg["model"]["type"] == "forward_only":
        reverse_observed = np.zeros_like(forward, dtype=float)
        total = forward
    else:
        reverse_intrinsic = reverse_shock(
            ivar,
            params["f0_rev"],
            params["nua0_rev"],
            params["num0_rev"],
            params["nuc0_rev"],
            k,
            cfg["burst"]["t0_rev"],
            p,
        )
        tau_abs_fs = forward_shock_absorption_tau(
            ivar,
            params["nua_0"],
            params["num_0"],
            params["nuc_0"],
            k,
            t0,
            p=p,
        )
        reverse_observed = reverse_intrinsic * np.exp(-tau_abs_fs)
        total = forward + reverse_observed

    return {
        "total": total,
        "forward": forward,
        "reverse": reverse_observed,
    }

def make_model(cfg):
    k = cfg["model"]["k"]
    p = cfg["model"]["p"]
    t0 = cfg["burst"]["t0"]
    t0_rev = cfg["burst"]["t0_rev"]

    if cfg["model"]["type"] == "forward_only":
        def model(theta, ivar):
            params = parameter_dict_from_theta(cfg, theta)
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
            params = parameter_dict_from_theta(cfg, theta)
        
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
    
    log_err = yerr / ((ydata + eps) * np.log(10))  # propagated uncertainty in log10 flux
    if not np.all(np.isfinite(log_err)) or np.any(log_err <= 0):
        return -np.inf

    log_like = -0.5 * np.sum(((log_data - log_model) / log_err) ** 2)

    if not np.isfinite(log_like):
        return -np.inf
    return log_like 


def log_probability(theta, model, keys, xdata, ydata, yerr, bounds):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    physical_theta = theta_to_physical(theta, keys)
    return lp + log_likelihood(physical_theta, model, xdata, ydata, yerr)


# ---------- MAIN SAMPLER ----------

def run_mcmc(cfg, xdata, ydata, yerr, nwalkers=None, nsteps=None):
    keys, p0, bounds = build_param_vector(cfg)
    model = make_model(cfg)

    ndim = len(p0)
    if ndim == 0:
        raise ValueError("❌ All fit parameters are fixed; at least one parameter must have non-equal bounds for MCMC.")

    fit_cfg = cfg.get("fit", {})
    nwalkers = int(fit_cfg.get("nwalkers", nwalkers or max(32, 2 * ndim + 2)))
    if nwalkers < 2 * ndim:
        nwalkers = 2 * ndim

    mode = fit_cfg.get("mcmc_mode", "adaptive")
    burn_in = int(fit_cfg.get("burn_in", 1000))
    fixed_steps = int(fit_cfg.get("nsteps", nsteps or 2000))
    max_steps = int(fit_cfg.get("max_steps", 50000))
    check_interval = int(fit_cfg.get("check_interval", 1000))
    autocorr_ratio = float(fit_cfg.get("autocorr_ratio", 50))
    autocorr_tol = float(fit_cfg.get("autocorr_tol", 0.02))

    pos = []
    for _ in range(nwalkers):
        trial = []
        for val, (low, high) in zip(p0, bounds):
            spread = 0.01 * (high - low) if high != low else 1e-6
            trial.append(np.clip(val + spread * np.random.randn(), low, high))
        pos.append(trial)
    
    pos = np.array(pos)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(model, keys, xdata, ydata, yerr, bounds),
    )

    print(f"Burn-in: {burn_in} steps, {nwalkers} walkers, {ndim} parameters")
    state = sampler.run_mcmc(pos, burn_in, progress=True)
    sampler.reset()

    diagnostics = {
        "mode": mode,
        "burn_in": burn_in,
        "converged": False,
        "tau_max": np.nan,
        "thin": 1,
    }

    if mode == "fixed":
        print(f"Production: {fixed_steps} fixed steps")
        sampler.run_mcmc(state, fixed_steps, progress=True)
    else:
        print(f"Production: adaptive, max {max_steps} steps")
        old_tau = None
        for _ in sampler.sample(state, iterations=max_steps, progress=True):
            if sampler.iteration % check_interval:
                continue
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except Exception:
                print(f"  step {sampler.iteration:6d}: autocorr not available yet")
                continue

            tau_max = float(np.max(tau))
            long_enough = sampler.iteration > autocorr_ratio * tau_max
            stable = (
                old_tau is not None
                and np.all(np.abs(old_tau - tau) / tau < autocorr_tol)
            )
            diagnostics["tau_max"] = tau_max
            print(
                f"  step {sampler.iteration:6d}: "
                f"tau_max={tau_max:.1f} "
                f"{'converged' if long_enough and stable else '...'}"
            )

            if long_enough and stable:
                diagnostics["converged"] = True
                break
            old_tau = tau

    acceptance = float(np.mean(sampler.acceptance_fraction))
    diagnostics["acceptance_fraction"] = acceptance
    diagnostics["production_steps"] = sampler.iteration
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        diagnostics["tau_max"] = float(np.max(tau))
        diagnostics["thin"] = max(1, int(0.5 * diagnostics["tau_max"]))
    except Exception:
        pass
    sampler.grbfit_diagnostics = diagnostics

    print(
        "Sampler diagnostics: "
        f"steps={diagnostics['production_steps']}, "
        f"acceptance={acceptance:.3f}, "
        f"tau_max={diagnostics['tau_max']:.1f}, "
        f"thin={diagnostics['thin']}, "
        f"converged={diagnostics['converged']}"
    )

    return keys, sampler
