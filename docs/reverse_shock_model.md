# Reverse-Shock Model

The reverse-shock component implemented in `grbfit.models.reverse_shock`
supports both the relativistic thick-shell and Newtonian thin-shell
reverse shocks from Table 5 of the reference table in this repository.
The thick-shell model remains the default for backward compatibility.

Select the model in `config.yaml` with:

```yaml
model:
  type: forward_reverse
  reverse_shell: thin  # thick or thin; defaults to thick
  fit_g: true          # only used by the thin-shell model
```

For a thin shell, the allowed Lorentz-factor index is

```text
g_min = (3 - k) / 2
g_max = (7 - 2k) / 2
```

When `fit_g` is false, `g` is fixed to the midpoint of this interval.
When `fit_g` is true, that midpoint is the initial value and the interval
provides the fit bounds. Users do not need to add `g` to `initial_guess`
or `bounds`. The supported environments are `k: 0` and `k: 2`, giving
fixed midpoint values `g = 2.5` and `g = 1.0`, respectively.

The fitted reverse-shock parameters are:

- `f0_rev`: reverse-shock `Fnu,max` at `t0_rev`
- `nua0_rev`: `nu_a` at `t0_rev`
- `num0_rev`: `nu_m` at `t0_rev`
- `nuc0_rev`: `nu_c` at `t0_rev`
- `g`: thin-shell Lorentz-factor index, when `fit_g` is true

Both shell models support these spectral orderings:

- `nu_a < nu_m < nu_c`
- `nu_a < nu_c < nu_m`
- `nu_m < nu_a < nu_c`

Orderings with `nu_a > nu_c` are not implemented and raise `ValueError`
instead of silently extrapolating.

## Thick-Shell Temporal Evolution

For every supported thick-shell reverse-shock regime, Table 5 gives:

```text
Fnu,max: -(47 - 10k) / (12 * (4 - k))
nu_m:    -(73 - 14k) / (12 * (4 - k))
nu_c:    -(73 - 14k) / (12 * (4 - k))
```

For low-absorption orderings, `nu_a < nu_m < nu_c` and
`nu_a < nu_c < nu_m`, the absorption break evolves as:

```text
nu_a: -(32 - 7k) / (15 * (4 - k))
```

For the self-absorbed slow-cooling ordering, `nu_m < nu_a < nu_c`,
the absorption break evolves as:

```text
nu_a: -[p * (73 - 14k) + 2 * (67 - 14k)]
       / [12 * (4 - k) * (p + 4)]
```

## Thin-Shell Temporal Evolution

For every supported thin-shell reverse-shock regime, Table 5 gives:

```text
Fnu,max: -(11g + 12) / [7 * (2g + 1)]
nu_m:    -3 * (5g + 8) / [7 * (2g + 1)]
nu_c:    -3 * (5g + 8) / [7 * (2g + 1)]
```

For the two low-absorption orderings:

```text
nu_a: -3 * (11g + 12) / [35 * (2g + 1)]
```

For the self-absorbed slow-cooling ordering:

```text
nu_a: -[3p * (5g + 8) + 8 * (4g + 5)]
       / [7 * (2g + 1) * (p + 4)]
```

## Spectral Regimes

The code is organized around the spectral ordering. Each regime uses a
triple smoothly broken power law with the break order and spectral
indices shown below.

### `nu_a < nu_m < nu_c`

Break order:

```text
nu_a, nu_m, nu_c
```

Spectral indices:

```text
2, 1/3, -(p - 1)/2, -p/2
```

The normalization passed to the smoothly broken power law is the flux at
`nu_a`:

```text
F(nu_a) = Fnu,max * (nu_a / nu_m)^(1/3)
```

### `nu_a < nu_c < nu_m`

Break order:

```text
nu_a, nu_c, nu_m
```

Spectral indices:

```text
2, 1/3, -1/2, -p/2
```

The normalization passed to the smoothly broken power law is the flux at
`nu_a`:

```text
F(nu_a) = Fnu,max * (nu_a / nu_c)^(1/3)
```

### `nu_m < nu_a < nu_c`

Break order:

```text
nu_m, nu_a, nu_c
```

Spectral indices:

```text
2, 5/2, -(p - 1)/2, -p/2
```

The normalization passed to the smoothly broken power law is the flux at
`nu_m`, following the same optically thick normalization convention used
in the forward-shock branch:

```text
F(nu_m) = Fnu,max * (nu_m / nu_a)^3
```

## Regime Transitions

If the reverse shock starts in `nu_a < nu_m < nu_c`, `nu_m` decreases
faster than `nu_a`, so the model can transition to
`nu_m < nu_a < nu_c`.

The crossing time is:

```text
t_cross = t0_rev * (num0_rev / nua0_rev)^(1 / (b_a - b_m))
```

where `b_a` is the low-absorption `nu_a` temporal index and `b_m` is the
`nu_m` temporal index.

For times after the crossing, the self-absorbed branch is evaluated with
breaks normalized at `t_cross`. The flux is then renormalized at the
same observing frequency:

```text
F_after(t, nu) = F_before(t_cross, nu)
                 * raw_after(t, nu)
                 / raw_after(t_cross, nu)
```

This is the same continuity idea used by the forward-shock model, but
the reverse-shock implementation keeps the branch logic grouped by
spectral regime.

## Code Orientation

- `reverse_shock_break_frequencies()` returns the time-dependent
  `(nu_a, nu_m, nu_c)` values for plotting and detectability checks.
- `_reverse_shock_temporal_indices()` selects the thick- or thin-shell
  Table 5 temporal indices.
- `_reverse_shock_spectrum()` contains the spectral shape and
  normalization for each supported ordering.
- `_reverse_shock_branch_state()` chooses the active regime at a given
  time and handles the `nu_a/nu_m` transition.
