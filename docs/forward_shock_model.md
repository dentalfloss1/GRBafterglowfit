# Forward-Shock Model

The forward-shock component is evaluated by
`grbfit.models.forward_shock_flux`. The older name `theory_bigsbpl`
remains as a compatibility alias.

The implemented non-jet model is the adiabatic ultra-relativistic
forward shock from van der Horst Table 2.8 and Table 2.10 in
`52659_proefschriftajvdhorst.pdf`.

The fitted forward-shock parameters are:

- `f0`: forward-shock `Fnu,max` at `t0`
- `nua_0`: `nu_a` at `t0`
- `num_0`: `nu_m` at `t0`
- `nuc_0`: `nu_c` at `t0`

The model also uses fixed `k`, fixed `p`, and optional `t_j`.

## Supported Non-Jet Regimes

The non-jet path chooses the starting spectral regime from the break
ordering at `t0`.

Supported orderings:

- `nu_a < nu_m < nu_c`
- `nu_a < nu_c < nu_m`
- `nu_m < nu_a < nu_c`

Unsupported orderings raise `ValueError` instead of silently
extrapolating.

## Temporal Evolution

For all supported adiabatic relativistic regimes:

```text
Fnu,max: -k / [2(4 - k)]
nu_m:    -3/2
nu_c:    -(4 - 3k) / [2(4 - k)]
```

The absorption break depends on the ordering:

```text
nu_a, fast cooling, nu_a < nu_c < nu_m:
  -(10 + 3k) / [5(4 - k)]

nu_a, slow cooling, nu_a < nu_m < nu_c:
  -3k / [5(4 - k)]

nu_a, self-absorbed slow cooling, nu_m < nu_a < nu_c:
  -[12p + 8 - 3pk + 2k] / [2(4 - k)(p + 4)]
```

## Spectral Shapes

Each regime is evaluated with a triple smoothly broken power law.

### `nu_a < nu_m < nu_c`

Break order:

```text
nu_a, nu_m, nu_c
```

Spectral slopes:

```text
2, 1/3, -(p - 1)/2, -p/2
```

The normalization passed to the smoothly broken power law is:

```text
F(nu_a) = Fnu,max * (nu_a / nu_m)^(1/3)
```

### `nu_a < nu_c < nu_m`

Break order:

```text
nu_a, nu_c, nu_m
```

Spectral slopes:

```text
2, 1/3, -1/2, -p/2
```

The normalization passed to the smoothly broken power law is:

```text
F(nu_a) = Fnu,max * (nu_a / nu_c)^(1/3)
```

### `nu_m < nu_a < nu_c`

Break order:

```text
nu_m, nu_a, nu_c
```

Spectral slopes:

```text
2, 5/2, -(p - 1)/2, -p/2
```

The normalization passed to the smoothly broken power law is:

```text
F(nu_m) = Fnu,max * (nu_m / nu_a)^3
```

## Regime Transitions

The non-jet implementation scales later branches from physical crossing
times.

For fast-to-slow cooling:

```text
nu_m(t_cross) = nu_c(t_cross)
```

The slow-cooling branch starts from the break values at that crossing.

For the slow-cooling self-absorption transition:

```text
nu_a(t_cross) = nu_m(t_cross)
```

The self-absorbed branch starts from the break values at that crossing.

At each transition, the new branch is renormalized at the transition time
and observing frequency:

```text
F_after(t, nu) = F_before(t_cross, nu)
                 * raw_after(t, nu)
                 / raw_after(t_cross, nu)
```

Because the branch break values are scaled from the physical crossing,
this is a continuity safeguard rather than a frequency-dependent
correction that changes the late-branch spectral slopes.

## Jet Branch

If `t_j` is supplied, the jet break is handled as another event in the
same reference-time evolution used for spectral transitions. The break
values and `Fnu,max` are evolved to `t_j` with the active pre-jet
indices, then the post-jet temporal indices are applied from `t_j`
forward.

For low-absorption branches, `nu_a < nu_m < nu_c` and
`nu_a < nu_c < nu_m`, the post-jet temporal indices are:

```text
Fnu,max: -1
nu_a:    -1/5
nu_m:    -2
nu_c:     0
```

For the self-absorbed slow-cooling branch, `nu_m < nu_a < nu_c`, the
post-jet temporal indices are:

```text
Fnu,max: -1
nu_a:    -2(p + 1) / (p + 4)
nu_m:    -2
nu_c:     0
```

If `nu_a` and `nu_m` cross after the jet break, the crossing time is
computed from the break values at `t_j`, not from the original `t0`
normalization. The self-absorbed branch is then anchored at that
physical crossing.

## Reverse-Shock Absorption

For `model.type: forward_reverse`, the forward-shock absorption screen
for reverse-shock photons is optional. By default it is not applied:

```yaml
model:
  apply_fs_absorption: false
```

Set `apply_fs_absorption: true` to evaluate:

```text
F_total(t, nu) = F_FS(t, nu) + F_RS(t, nu) * exp[-tau_FS(t, nu)]
```

When omitted from older config files, `apply_fs_absorption` is treated
as `false`.
