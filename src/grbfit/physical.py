"""Transform observable draws using van der Horst (2007), Tables 2.6--2.8.

The prescription, constants, normalization mapping, and domain are documented
in docs/physical_parameters.md. This module does not refit or reweight draws.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from grbfit.artifacts import load_posterior_samples


# Gaussian cgs; fixed here so saved prescription versions remain reproducible.
C = 2.99792458e10
QE = 4.80320471257e-10
ME = 9.1093837139e-28
MP = 1.67262192595e-24
MPC = 3.0856775814913673e24
DAY = 86400.0
WIND_A = 5e11  # rho = A r^-2, A = WIND_A * A_star, in g cm^-1
PRESCRIPTION = "van_der_horst_2007_cgs_v1"
OBSERVABLES = ("f0", "nua_0", "num_0", "nuc_0")
REGIMES = ("slow", "fast", "self_absorbed_slow")


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def luminosity_distance_cm(z, h0=70.0, omega_m=0.3):
    """Flat matter + Lambda cosmology (radiation neglected), in cm."""
    _positive("redshift", z)
    _positive("H0", h0)
    if not np.isfinite(omega_m) or not 0 <= omega_m <= 1:
        raise ValueError("Omega_m must lie between 0 and 1")
    integral = quad(lambda x: 1 / np.sqrt(omega_m * (1 + x)**3 + 1 - omega_m), 0, z)[0]
    return (1 + z) * (C / 1e5) / h0 * integral * MPC


def _relations(p, k, time_days, z, distance_cm, regime):
    """Return ln prefactors and exponents for [Fmax/Jy, nua/Hz, num/Hz, nuc/Hz].

    Columns of the exponent matrix are [E/erg, n/cm^-3 or A/(g cm^-1),
    epsilon_e, epsilon_B]. Coefficients are evaluated in log space.
    """
    if k not in (0, 2):
        raise ValueError("Physical conversion supports only k=0 or k=2")
    if not np.isfinite(p) or p <= 2:
        raise ValueError("The electron prescription requires p > 2")
    if regime not in REGIMES:
        raise ValueError(f"Unsupported spectral regime: {regime}")
    for name, value in (("time_days", time_days), ("redshift", z), ("distance_cm", distance_cm)):
        _positive(name, value)
    # E = alpha*pi*rho(R)*R^3*Gamma^2*c^2; R = beta*Gamma^2*c*t/(1+z).
    alpha, beta = 16 / (17 - 4 * k), 4 - k
    l2, l3, la, lb, lp, lx, lq, le, lm, lc = np.log(
        [2, 3, alpha, beta, np.pi, 0.85, QE, ME, MP, C]
    )
    lt, lz, ld = np.log([time_days * DAY, 1 + z, distance_cm])
    if k == 0:
        f = np.log(4 / 3) + .5*l2 - la - .5*lp + lx + 3*lq - le - .5*lm - 3*lc - 2*ld + lz
        c = np.log(81 / 8192) + .5*l2 + .5*la + 1.5*lb - 2*lp - 7*lq + 5*le - lm + 8.5*lc - .5*lt - .5*lz
        fexp, cexp = [1, .5, 0, .5], [-.5, -1, 0, -1.5]
    else:
        f = np.log(4 / 3) + .5*l2 - .5*la - .5*lb + lx + 3*lq - le - lm - 2.5*lc - 2*ld - .5*lt + 1.5*lz
        c = np.log(81 / 8192) + .5*l2 - .5*la + 2.5*lb - 3*lp - 7*lq + 5*le + 7.5*lc + .5*lt - 1.5*lz
        fexp, cexp = [.5, 1, 0, .5], [.5, -2, 0, -1.5]
    m = (1.5*l2 - .5*la - 1.5*lb - lp - 2*lx
         + 2*np.log((p - 2) / (p - 1)) + lq - 3*le + 2*lm - 2.5*lc - 1.5*lt + .5*lz)
    if regime == "slow":
        a = (l2 + .8*l3 + 1.6*lx - np.log(p - 2) + 1.6*np.log(p - 1)
             - .6*np.log(p + 2/3) + .6*np.log(p + 2) + 1.6*lq)
        if k == 0:
            a += -.2*la + .2*lp - lm - lc - lz
            aexp = [.2, .6, -1, .2]
        else:
            a += .4*la - .6*lb + .8*lp - 1.6*lm - .4*lc - .6*lt - .4*lz
            aexp = [-.4, 1.2, -1, .2]
    elif regime == "fast":
        a = 5.6*l2 - .6*l3 + .6*lx + 5.6*lq - 4*le
        if k == 0:
            a += -.7*la - 1.5*lb + .1*lp + .5*lm - 6.5*lc - .5*lt - .5*lz
            aexp = [.7, 1.1, 0, 1.2]
        else:
            a += .4*la - 2.6*lb + 1.2*lp - .6*lm - 5.4*lc - 1.6*lt + .6*lz
            aexp = [-.4, 2.2, 0, 1.2]
    else:
        d = p + 4
        a = ((9*p - 22)/(6*d)*l2 + 8/(3*d)*l3 - 2*(p+2)/d*lx
             + 2/d*gammaln(p/2 + 1/3) + 2*(p-1)/d*np.log(p-2)
             - 2*(p-2)/d*np.log(p-1) + 2/d*np.log(p+2)
             + (p+6)/d*lq - (3*p+2)/d*le)
        if k == 0:
            a += (-(p+2)/(2*d)*la - (3*p+2)/(2*d)*lb - (p+2)/d*lp
                  + 2*(p-1)/d*lm - (5*p+10)/(2*d)*lc
                  - (3*p+2)/(2*d)*lt + (p-6)/(2*d)*lz)
            aexp = [(p+2)/(2*d), 2/d, 2*(p-1)/d, (p+2)/(2*d)]
        else:
            a += (-(p-2)/(2*d)*la - 3*(p+2)/(2*d)*lb - p/d*lp
                  + 2*(p-2)/d*lm - (5*p+6)/(2*d)*lc
                  - 3*(p+2)/(2*d)*lt + (p-2)/(2*d)*lz)
            aexp = [(p-2)/(2*d), 4/d, 2*(p-1)/d, (p+2)/(2*d)]
    return np.array([f - np.log(1e-23), a, m, c]), np.array([fexp, aexp, [.5, 0, 2, .5], cexp])


def predict_observables(energy_erg, density, epsilon_e, epsilon_b, *, p, k,
                        time_days, z, distance_cm, regime):
    """Physical Fmax (Jy), nua, num, nuc (GHz); density is n or A_star."""
    values = np.array([energy_erg, density, epsilon_e, epsilon_b], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Physical inputs must be finite and positive")
    if k == 2:
        values[1] *= WIND_A
    intercept, powers = _relations(p, k, time_days, z, distance_cm, regime)
    return np.exp(intercept + powers @ np.log(values)) / [1, 1e9, 1e9, 1e9]


def jet_half_angle(energy_erg, density, time_days, z, k):
    """Gamma(t_j)*theta_j=1 with the same dynamics as the spectral conversion."""
    if k not in (0, 2):
        raise ValueError("Jet conversion supports only k=0 or k=2")
    for name, value in (("energy", energy_erg), ("density", density), ("t_j", time_days), ("redshift", z)):
        _positive(name, value)
    rho0 = density * (MP if k == 0 else WIND_A)
    log_theta = (np.log(16*np.pi/(17-4*k)) + np.log(rho0) + (5-k)*np.log(C)
                 + (3-k)*np.log((4-k)*time_days*DAY/(1+z)) - np.log(energy_erg)) / (8-2*k)
    return float(np.exp(log_theta))


def derive_physical_posterior(samples, keys, metadata, *, z=None, distance_cm=None,
                              h0=70.0, omega_m=0.3, allow_fixed_observables=False):
    """Return samples, names, per-row statuses, and a JSON-safe summary.

    Failed-domain rows contain NaNs. Algebraic solutions outside the energy
    fraction domain remain saved but are excluded from the conditional summary.
    """
    cfg = metadata["config"]
    k, t0 = cfg["model"]["k"], cfg["burst"]["t0"]
    if k not in (0, 2):
        raise ValueError("Physical conversion supports only k=0 or k=2")
    _positive("t0", t0)
    z = cfg["burst"].get("z") if z is None else z
    if z is None:
        raise ValueError("Redshift is required; supply --redshift or burst.z in the fit")
    _positive("redshift", z)
    distance_source = "supplied" if distance_cm is not None else "flat_LambdaCDM"
    if distance_cm is None:
        distance_cm = luminosity_distance_cm(z, h0, omega_m)
    _positive("luminosity distance", distance_cm)
    fixed = metadata["fixed_params"]
    fixed_observables = [name for name in OBSERVABLES if name in fixed]
    if fixed_observables and not allow_fixed_observables:
        raise ValueError("Fixed observables: " + ", ".join(fixed_observables)
                         + ". Use --allow-fixed-observables for a conditional calculation.")
    for name in (*OBSERVABLES, "p"):
        if name not in keys and name not in fixed:
            raise ValueError(f"Missing required observable: {name}")
    names = ["E_k_iso", "n" if k == 0 else "A_star", "epsilon_e", "epsilon_B"]
    has_jet = "t_j" in keys or "t_j" in fixed
    if has_jet:
        names += ["theta_j_rad", "theta_j_deg", "E_k_beaming_corrected"]
    derived = np.full((len(samples), len(names)), np.nan)
    status = np.full(len(samples), "unsupported_ordering", dtype="U32")
    regimes = np.full(len(samples), "unsupported", dtype="U32")
    for i, row in enumerate(samples):
        params = {**fixed, **dict(zip(keys, row))}
        f, a, m, c = [params[name] for name in OBSERVABLES]
        p = params["p"]
        if not np.all(np.isfinite([f, a, m, c, p])) or min(f, a, m, c) <= 0:
            status[i] = "invalid_observables"
            continue
        if p <= 2:
            status[i] = "unsupported_p"
            continue
        if has_jet and (not np.isfinite(params["t_j"]) or params["t_j"] <= 0):
            status[i] = "invalid_jet_time"
            continue
        if has_jet and t0 >= params["t_j"]:
            status[i] = "reference_not_prejet"
            continue
        if a < m < c:
            regime = "slow"
        elif a < c < m:
            regime = "fast"
        elif m < a < c:
            regime = "self_absorbed_slow"
        else:
            continue
        regimes[i] = regime
        log_obs = np.log([f, a, m, c]) + np.log([1, 1e9, 1e9, 1e9])
        if regime == "self_absorbed_slow":
            # Match the fitted optically thin asymptote to Table 2.8.
            log_obs[0] += (2-p)/2 * (np.log(m) - np.log(a))
        intercept, powers = _relations(p, k, t0, z, distance_cm, regime)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            solution = np.exp(np.linalg.solve(powers, log_obs - intercept))
        if k == 2:
            solution[1] /= WIND_A
        if np.any(~np.isfinite(solution)) or np.any(solution <= 0):
            status[i] = "numerical_failure"
            continue
        derived[i, :4] = solution
        status[i] = "valid" if solution[2] + solution[3] <= 1 else "unphysical_energy_fractions"
        if has_jet:
            theta = jet_half_angle(solution[0], solution[1], params["t_j"], z, k)
            if not np.isfinite(theta) or not 0 < theta < np.pi/2:
                status[i] = "jet_outside_domain"
                continue
            derived[i, 4:] = [theta, np.degrees(theta), solution[0]*2*np.sin(theta/2)**2]
    valid = status == "valid"
    units = dict(zip(names, ["erg", "cm^-3" if k == 0 else "dimensionless", "dimensionless",
                             "dimensionless", "rad", "deg", "erg"]))
    summary = {}
    if np.any(valid):
        quantiles = np.percentile(derived[valid], [16, 50, 84], axis=0)
        for j, name in enumerate(names):
            low, median, high = quantiles[:, j]
            summary[name] = dict(median=float(median), q16=float(low), q84=float(high), unit=units[name])
    warnings = []
    if fixed_observables:
        warnings.append("Results are conditional on fixed observables: " + ", ".join(fixed_observables))
    if metadata.get("sampler", {}).get("converged") is not True:
        warnings.append("Input sampler convergence was not established.")
    if not np.all(valid):
        warnings.append("Intervals are conditional on valid rows; inspect status_counts and excluded fraction.")
    report = {
        "format": "grbfit.physical_posterior", "format_version": 1,
        "prescription": PRESCRIPTION,
        "reference": "van der Horst (2007), Broadband View of Blast Wave Physics, Tables 2.6, 2.7, 2.8",
        "assumptions": {"xi_e": 1.0, "Y": 0.0, "hydrogen_mass_fraction": 0.7,
                        "dynamics": "adiabatic_ultrarelativistic", "component": "forward_shock",
                        "jet": "on_axis_top_hat_Gamma_theta_1", "energy": "kinetic",
                        "energy_fraction_domain": "epsilon_e + epsilon_B <= 1",
                        "beaming_energy": "two_sided: E_iso*(1-cos(theta_j))"},
        "redshift": z, "distance_cm": distance_cm, "distance_source": distance_source,
        "cosmology": {"H0_km_s_Mpc": h0, "Omega_m": omega_m, "radiation": False}
                     if distance_source == "flat_LambdaCDM" else None,
        "reference_time_days": t0, "k": k, "parameter_units": units,
        "fixed_observables": fixed_observables, "input_metadata": metadata,
        "total_draws": len(samples), "valid_draws": int(valid.sum()),
        "excluded_fraction": float(1 - valid.mean()),
        "status_counts": {str(s): int(np.sum(status == s)) for s in np.unique(status)},
        "regime_counts": {str(r): int(np.sum(regimes == r)) for r in np.unique(regimes)},
        "summary": summary, "warnings": warnings,
    }
    return derived, names, status, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive forward-shock physical parameters from saved joint draws.")
    parser.add_argument("posterior", nargs="?", default="posterior_samples.npz")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--redshift", type=float, help="Override the archived redshift")
    parser.add_argument("--distance-cm", type=float, help="Supply luminosity distance instead of computing it")
    parser.add_argument("--h0", type=float, default=70.0)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--allow-fixed-observables", action="store_true")
    args = parser.parse_args(argv)
    try:
        samples, keys, metadata = load_posterior_samples(args.posterior)
        derived, names, status, report = derive_physical_posterior(
            samples, keys, metadata, z=args.redshift, distance_cm=args.distance_cm,
            h0=args.h0, omega_m=args.omega_m, allow_fixed_observables=args.allow_fixed_observables,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report["input_file"] = str(Path(args.posterior).resolve())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    np.savez_compressed(output / "physical_samples.npz", samples=derived, param_keys=np.asarray(names),
                        source_sample_index=np.arange(len(samples)), status=status,
                        metadata_json=np.asarray(encoded))
    (output / "physical_summary.json").write_text(encoded + "\n")
    print(f"Physical conversion: {report['valid_draws']}/{report['total_draws']} valid draws")
    for name, entry in report["summary"].items():
        print(f"{name}: {entry['median']:.6g} [{entry['q16']:.6g}, {entry['q84']:.6g}] {entry['unit']}")
    for message in report["warnings"]:
        print(f"Warning: {message}")
    print(f"Saved physical_samples.npz and physical_summary.json in {output}")
    if not report["valid_draws"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
