If you use this code, please cite or drop a link to this GitHub page. 

Based on modeling code originally written for Chastain, S. I., et al. (2026). “GRB 240205B: A Reverse Shock Detected in Rapid Response Radio Observations.” arXiv:2603.19047 [astro-ph.HE]. Submitted to Publications of the Astronomical Society of Australia.

This code uses the standard GRB afterglow models in either an ISM-like ($k=0$) or stellar wind-like ($k=2$) medium to give the expected temporal and spectral indices. These are then fit to the data with $p$ fixed by default or fitted within user-supplied bounds. The outputs are the observational parameters, not physical parameters: $F_0$, $\nu_a$, $\nu_m$, $\nu_c$, etc. This is done in order to provide fits to data even when there is not enough data to find the physical parameters.

To fit `p`, set `fit.initial_guess.p` to a starting value and give `fit.bounds.p` distinct lower and upper values, for example `[2.0, 3.0]`. The template fixes `p` at 2.2 with bounds `[2.2, 2.2]`. Existing configs with `model.p` and no `fit` entry for `p` continue to use that fixed value. If both initial values are provided, they must agree.

To exclude frequencies below 50 GHz from fitting, add `min_freq` to the `fit`
section of `config.yaml`:

```yaml
fit:
  min_freq: 50
```

This cutoff uses observed frequencies in GHz, regardless of redshift. Points
at exactly 50 GHz remain eligible for fitting, subject to the other data cuts.
Lower-frequency detections and upper limits remain on the posterior model plot
for reference; excluded detections use open markers. Omit `min_freq` or set it
to `null` to disable the cutoff.

Each run saves `residuals.png` for fitted detections and
`residuals_non_fitted.png` for detections excluded by the fitting cuts, when
valid excluded detections are available. Both use the same posterior-median
model, evaluated at each observation's time and frequency, with no refitting.
Residuals are `(log10(observed) - log10(model)) / sigma_log10`, using
`sigma_log10 = sqrt(err² + rms²) / (flux * ln(10))`. Positive residuals mean
the data lie above the model; values near zero indicate agreement. These
uncertainties describe the measurements and do not include posterior model
uncertainty. Upper limits remain on `posterior_models.png` and are omitted
from both residual plots.

Upper limits now constrain fits by default, including when `fit.fit_upper_limits`
is omitted from an existing configuration. Set `fit_upper_limits: false` in the
`fit` section to restore detection-only fitting. Existing fits may therefore
produce different posteriors after this update.

Rows with `flux > 0` and `err > 0` are detections; other rows retain the existing
upper-limit classification. For selected upper limits, `sigma = |rms|` and
`limit = 3 * sigma`, in the input's microJy units. Sentinel `flux` and `err`
values are not treated as measurements. Times, frequencies, and `|rms|` must
be finite and positive. Limits follow the same selection cuts as detections,
and at least one detection must remain.

Assuming independent Gaussian measurement noise in linear flux, each limit
adds `log Phi((limit - model_flux) / sigma)` to the likelihood, where `Phi` is
the standard normal CDF. Predictions well above a limit are strongly penalized;
predictions well below it receive little penalty. The detection likelihood
remains in log-flux space. Extinction corrections apply to both kinds of data.

The fit report lists whether limits are enabled, their count, and their summed
log-likelihood at the posterior-median parameters. Chi-square, reduced chi-square,
degrees of freedom, AIC, and BIC remain **detection-only diagnostics**, including
the existing columns in `fit_summary.csv`; they do not describe the combined
censored likelihood or provide full-likelihood model selection scores.

## Command line

Run `grbfit` from a directory containing `config.yaml`. If that file is absent,
the command creates a template and exits.

```sh
grbfit --help
grbfit --quick
```

`--quick` uses fixed-length MCMC with 100 burn-in steps and 300 production
steps for an exploratory fit. It overrides the sampler settings in memory for
that run; `config.yaml` is not changed. Run `grbfit` without the flag for the
configured sampler settings.

### Fitting extinction-affected bands above the cutoff

By default, bands at or above `fit.max_rest_freq` remain excluded. To fit
non-X-ray bands with at least three detections after the time and minimum
frequency cuts, add:

```yaml
fit:
  fit_above_max_rest_freq: true
  extinction_correction_bounds: [1, 1000]
  extinction_correction_initial: 2
```

This requires a defined `burst.z` and a finite positive `fit.max_rest_freq`
(in rest-frame GHz). Observed frequencies within a 1% span are grouped into a band,
pooling instruments and detections. Groups are built in increasing frequency
order, with the highest frequency at most 1.01 times the lowest; adjacent
matches cannot chain into a wider band. For example, 4.56e14 Hz and 4.6e14 Hz
(456000 and 460000 GHz in the input files) share one correction. Original
frequencies are retained when evaluating the intrinsic model. Grouping applies
to the non-X-ray data above the cutoff after the other cuts. Upper limits do not
count toward eligibility, but limits in eligible bands are fitted when
`fit_upper_limits` is enabled.
Bands with fewer than three detections remain plotted for reference. X-rays
remain controlled by `fit_xrt` and receive no extra correction.

Each eligible band gets an independent, time-independent correction `C >= 1`,
representing the upward correction to observed flux for extinction. The fit
compares the unchanged measurements and errors to `intrinsic_model / C`.
The prior is uniform in `log10(C)` within the shared bounds (default 1–1000),
with initial value 2. Bounds must satisfy `1 <= lower < initial < upper`.
This is a per-band correction, not an extinction-curve or redshift fit.

Parameters named `extinction_correction_1`, etc. are assigned in increasing
frequency order. The lowest frequency represents each group in
`fit.extinction_bands`, and all member frequencies are saved in
`fit.extinction_band_frequencies`. JSON parameter summaries include both
`observed_frequency_GHz` and `observed_frequencies_GHz`. Plots, residuals, and
exported-model evaluations apply the shared correction at every saved member
frequency, without interpolation to other frequencies. Older saved fits with
single-frequency corrections remain supported. The standardized summary CSV retains its
existing columns. Omitted or false `fit_above_max_rest_freq` preserves the
existing selection and model behavior.
