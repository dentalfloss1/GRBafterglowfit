If you use this code, please cite or drop a link to this GitHub page. 

Based on modeling code originally written for Chastain, S. I., et al. (2026). “GRB 240205B: A Reverse Shock Detected in Rapid Response Radio Observations.” arXiv:2603.19047 [astro-ph.HE]. Submitted to Publications of the Astronomical Society of Australia.

This code uses the standard GRB afterglow models in either an ISM-like ($k=0$) or stellar wind-like ($k=2$) medium to give the expected temporal and spectral indices. These are then fit to the data with $p$ fixed by default or fitted within user-supplied bounds. The outputs are the observational parameters, not physical parameters: $F_0$, $\nu_a$, $\nu_m$, $\nu_c$, etc. This is done in order to provide fits to data even when there is not enough data to find the physical parameters.

To fit `p`, set `fit.initial_guess.p` to a starting value and give `fit.bounds.p` distinct lower and upper values, for example `[2.0, 3.0]`. The template fixes `p` at 2.2 with bounds `[2.2, 2.2]`. Existing configs with `model.p` and no `fit` entry for `p` continue to use that fixed value. If both initial values are provided, they must agree.

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
