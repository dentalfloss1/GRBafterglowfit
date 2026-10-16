# Physical parameters from saved observable draws

Run the separate postprocessor after a fit:

```sh
grbfit-physical posterior_samples.npz --output-dir physical
# Equivalent without installing the new console entry point:
python -m grbfit.physical posterior_samples.npz --output-dir physical
```

The input is the stage-one joint posterior archive, not `model_fit.json`.
Each row is transformed separately, preserving its fitted `p`, break
frequencies, normalization, and jet-time correlations. This is a deterministic
transformation of the observable posterior under the assumptions below; it
does not impose new physical priors or run another MCMC. The original fit is
not modified. Reverse-shock and extinction nuisance parameters remain
marginalized through the joint input samples; only the forward-shock
observables enter the conversion.

## Inputs and reference time

The reference time is **`burst.t0`, in observer-frame days since the trigger**,
where the fitted forward-shock normalization and three break frequencies are
defined. It is already in the archive; no calendar trigger date is needed.
All four observables must be positive. Their input units are Jy and GHz.
The supported environments are `k=0` and `k=2`, and `p` must exceed 2.

Redshift comes from the archive's `burst.z`, or an explicit override:

```sh
grbfit-physical posterior_samples.npz --redshift 1.5
```

Luminosity distance defaults to a flat matter-plus-Lambda cosmology with
`H0=70 km/s/Mpc`, `Omega_m=0.3`, and radiation neglected. Override those with
`--h0` and `--omega-m`, or supply `--distance-cm` to use a distance from another
cosmology. Redshift is still required because frequencies and times need
redshift factors. Redshift and distance are treated as exact inputs; their
uncertainties are not propagated. Their adopted values and source are saved.

If any of `f0`, `nua_0`, `num_0`, or `nuc_0` was fixed, the command requires
`--allow-fixed-observables`. Such results are explicitly conditional on those
values. Fixed `p` and fixed `t_j` are allowed and recorded in the input
metadata. Fitting a break does not establish that it is observationally
constrained; inspect the original posterior and prior bounds.

For a jet fit, this version requires **`t0 < t_j` for each draw**. It does not
extrapolate post-jet reference observables back into the spherical phase.
For example, `t0=1` day and `t_j=5` days is supported. If some draws have a
jet time at or before `t0`, those rows receive `reference_not_prejet` status.
No jet angle or limit is inferred when `t_j` is absent.

## Physical prescription

The implementation uses the analytic cgs expressions in A. J. van der Horst
(2007), *Broadband View of Blast Wave Physics: A Study of Gamma-Ray Burst
Afterglows*, Tables 2.6 (ISM), 2.7 (wind), and 2.8 (spectral normalization).
The local reference used for Tables 2.6 and 2.7 was
`/home/sarahc/vanderhorstscalings.pdf`; the repository's
`52659_proefschriftajvdhorst.pdf` contains Table 2.8. Neither PDF is needed
at runtime. The thesis is listed in the
[NASA thesis bibliography](https://hesperia.gsfc.nasa.gov/rhessi/news-and-resources/theses/).

The saved prescription identifier is `van_der_horst_2007_cgs_v1`. Assumptions:

- Adiabatic, ultra-relativistic forward shock, constant energy and microphysics.
- All electrons accelerated, `xi_e=1`; synchrotron cooling only, `Y=0`.
- Hydrogen mass fraction `X=0.7`; the electron-energy prescription requires `p>2`.
- ISM density `n=rho/m_p` in cm^-3, or wind density
  `rho(r)=A r^-2`, with `A=5e11 A_star g/cm`.
- For a jet: an on-axis uniform top-hat jet with `Gamma(t_j) theta_j=1`.

All three fitted break orderings are supported, selected separately per row:
`nu_a < nu_m < nu_c`, `nu_a < nu_c < nu_m`, and
`nu_m < nu_a < nu_c`. The fast-cooling calculation uses the same single
absorption-break approximation as the fitter. It does not substitute the
additional fast-cooling absorption structure of other prescriptions.

`_relations` in `grbfit.physical` defines the four log-linear equations.
Their unknowns are `ln(E_k,iso/erg)`, `ln(n/cm^-3)` or `ln(A/(g/cm))`,
`ln(epsilon_e)`, and `ln(epsilon_B)`. Solving this four-by-four system for
each draw avoids an optimizer and its initial guesses. Frequencies are
converted to Hz for the equations. Fundamental constants are explicitly
versioned in the module rather than taken from an evolving external database.

For checking the normalization, the tables give the following rounded
values at `E=1e52 erg`, `n=1` or `A_star=1`, `epsilon_e=epsilon_B=0.1`,
`p=2.2`, `t=1 observer day`, `z=1`, and `D_L=1e28 cm`:

| Observable | ISM | Wind |
|---|---:|---:|
| Fnu,max (Jy) | 0.0213 | 0.0608 |
| nu_m (Hz) | 8.98e11 | 1.85e12 |
| nu_c (Hz) | 5.98e13 | 9.97e11 |
| nu_a, low-absorption slow (Hz) | 7.75e10 | 5.16e11 |
| nu_a, fast (Hz) | 1.25e9 | 9.23e10 |
| nu_a, self-absorbed slow (Hz) | 1.13e11 | 4.38e11 |

These rows test equation normalizations, not simultaneous regime validity:
not all absorption branches satisfy their ordering at these common inputs.
The implemented analytic cgs expressions agree within 1.5% with these rounded
values. The `nu_m` prefactor is `2 sqrt(2)` in both environments, consistent
with the numerical normalizations (the local ISM PDF has displaced typesetting
of the leading 2). The full `p` dependence is retained.

### Mapping the fitted normalization

For either low-absorption ordering, the fitted `f0` is the prescription's
asymptotic `Fnu,max`, rather than the smoothed spectrum evaluated at a break.
No additional smoothing correction is applied.

For `nu_m < nu_a < nu_c`, the current fitter uses
`A=f0*(nu_m/nu_a)^3` at its lowest break. Table 2.8 instead uses
`A=Fnu,max*(nu_m/nu_a)^((p+4)/2)`. Thus the postprocessor uses

```text
Fnu,max = f0 * (nu_m/nu_a)^((2-p)/2).
```

This matches the optically thin asymptote of the actual fitted spectrum.
It does not change any existing fit or model evaluation. The correction is
essential when `p` differs from 2.

### Jet angle

Use the same dynamical convention as the spectral coefficients:

```text
alpha = 16/(17-4k), beta = 4-k
rho0 = m_p*n (ISM), or A (wind)
E_iso = alpha*pi*rho0*R^(3-k)*Gamma^2*c^2
R = beta*Gamma^2*c*t_observer/(1+z)

theta_j = [alpha*pi*rho0*c^(5-k)
           * (beta*t_j/(1+z))^(3-k) / E_iso]^(1/(8-2k))
```

Here `t_j` is in seconds within the equation. This is a convention-dependent
jet half-opening angle, with the usual ISM `t_j^(3/8)` and wind `t_j^(1/4)`
scalings. It uses inferred **kinetic energy**, not prompt gamma-ray energy;
no prompt radiative efficiency is needed. Report angles in radians and degrees.
The additional beaming-corrected kinetic energy is the two-sided convention,
`E_k = E_k,iso*(1-cos(theta_j))`. This is an approximate relativistic jet
prescription, not a hydrodynamic jet or viewing-angle fit.

## Outputs and invalid draws

`physical_samples.npz` contains linear physical `samples`, ordered
`param_keys`, `source_sample_index`, a per-row `status`, and scalar JSON
`metadata_json`. Load with `numpy.load(..., allow_pickle=False)`. Row indices
map directly to the original archive. Columns are `E_k_iso`, `n` or `A_star`,
`epsilon_e`, `epsilon_B`, plus `theta_j_rad`, `theta_j_deg`, and
`E_k_beaming_corrected` when a jet time is present.

`physical_summary.json` contains the assumptions, prescription, units,
original metadata, counts by regime and status, excluded fraction, warnings,
and the 16th, 50th, and 84th percentiles of **valid rows only**. Repeated runs
replace these outputs in the selected directory. Input convergence diagnostics
are preserved; a warning accompanies runs without established convergence.

Rows outside the equation domain are retained as NaNs with explicit statuses:
`unsupported_p`, `unsupported_ordering`, `reference_not_prejet`,
`invalid_observables`, `invalid_jet_time`, or `numerical_failure`.
Algebraic solutions with `epsilon_e + epsilon_B > 1` remain in the archive
without clipping and are labeled `unphysical_energy_fractions`. Angles outside
`0 < theta_j < pi/2` receive `jet_outside_domain`; finite physical quantities
already computed remain saved. Neither criterion establishes that all other
model assumptions hold.

Intervals therefore describe the transformed draws **conditional on the
stated physical domain**; inspect the excluded fraction before interpreting
them. No draws are silently replaced, independently resampled, or reweighted.
If no valid rows remain, the diagnostic files are still written, the summary
is empty, and the command exits with status 1. Missing required inputs produce
an error before outputs are written.
