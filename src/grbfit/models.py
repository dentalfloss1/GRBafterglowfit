import numpy as np

RS_SLOW_COOLING = "nu_a < nu_m < nu_c"
RS_FAST_COOLING = "nu_a < nu_c < nu_m"
RS_SELF_ABSORBED_SLOW = "nu_m < nu_a < nu_c"

FS_SLOW_COOLING = "nu_a < nu_m < nu_c"
FS_FAST_COOLING = "nu_a < nu_c < nu_m"
FS_SELF_ABSORBED_SLOW = "nu_m < nu_a < nu_c"


def dsbpl(x,A,xb1,alpha1,alpha2,xb2,alpha3,s=0.2):
    """
    Multiplicative smoothly broken power law
    """

    x = np.asarray(x)
    alpha1 = -alpha1
    alpha2 = -alpha2
    alpha3 = -alpha3
    s1 = s
    s2 = s

    term1 = (x / xb1) ** (-alpha1)

    smooth1 = (1 + (x / xb1) ** (1.0 / s1)) ** ((alpha1 - alpha2) * s1)
    smooth2 = (1 + (x / xb2) ** (1.0 / s2)) ** ((alpha2 - alpha3) * s2)

    return A * term1 * smooth1 * smooth2
def tsbpl(x, A, xb1, xb2, xb3, alpha1, alpha2, alpha3, alpha4, s=0.2):
    x = np.asarray(x)

    alpha1 = -alpha1
    alpha2 = -alpha2
    alpha3 = -alpha3
    alpha4 = -alpha4

    term1 = (x / xb1) ** (-alpha1)

    smooth1 = (1 + (x / xb1) ** (1.0 / s)) ** ((alpha1 - alpha2) * s)
    smooth2 = (1 + (x / xb2) ** (1.0 / s)) ** ((alpha2 - alpha3) * s)
    smooth3 = (1 + (x / xb3) ** (1.0 / s)) ** ((alpha3 - alpha4) * s)

    return A * term1 * smooth1 * smooth2 * smooth3


def _forward_shock_relativistic_indices(k, p, regime):
    """Adiabatic ultra-relativistic FS temporal indices from van der Horst Table 2.10."""
    fmax = -k / (2 * (4 - k))
    num = -3 / 2
    nuc = -(4 - 3 * k) / (2 * (4 - k))

    if regime == FS_FAST_COOLING:
        nua = -(10 + 3 * k) / (5 * (4 - k))
    elif regime == FS_SLOW_COOLING:
        nua = -3 * k / (5 * (4 - k))
    elif regime == FS_SELF_ABSORBED_SLOW:
        nua = -(
            12 * p + 8 - 3 * p * k + 2 * k
        ) / (2 * (4 - k) * (p + 4))
    else:
        raise ValueError(f"Unsupported forward-shock spectral regime: {regime}")

    return fmax, nua, num, nuc


def _forward_shock_jet_indices(k, p, regime):
    """Post-jet temporal indices used by the legacy forward-shock implementation."""
    fmax = -1
    num = -2
    nuc = 0

    if regime in (FS_FAST_COOLING, FS_SLOW_COOLING):
        nua = -1 / 5
    elif regime == FS_SELF_ABSORBED_SLOW:
        nua = -(2 * (p + 1)) / (p + 4)
    else:
        raise ValueError(f"Unsupported forward-shock spectral regime: {regime}")

    return fmax, nua, num, nuc


def _forward_shock_temporal_indices(k, p, regime, jet_phase=False):
    if jet_phase:
        return _forward_shock_jet_indices(k, p, regime)
    return _forward_shock_relativistic_indices(k, p, regime)


def _forward_shock_spectrum(regime, nu, fnu_max, nua, num, nuc, p):
    if regime == FS_SLOW_COOLING:
        # nu_a < nu_m < nu_c
        f_at_nua = fnu_max * (nua / num) ** (1 / 3)
        return tsbpl(nu, f_at_nua, nua, num, nuc, 2, 1 / 3, -(p - 1) / 2, -p / 2)

    if regime == FS_FAST_COOLING:
        # nu_a < nu_c < nu_m
        f_at_nua = fnu_max * (nua / nuc) ** (1 / 3)
        return tsbpl(nu, f_at_nua, nua, nuc, num, 2, 1 / 3, -1 / 2, -p / 2)

    if regime == FS_SELF_ABSORBED_SLOW:
        # nu_m < nu_a < nu_c
        f_at_num = fnu_max * (num / nua) ** 3
        return tsbpl(nu, f_at_num, num, nua, nuc, 2, 5 / 2, -(p - 1) / 2, -p / 2)

    raise ValueError(f"Unsupported forward-shock spectral regime: {regime}")


def _forward_shock_initial_regime(nua0, num0, nuc0):
    if nua0 < num0 < nuc0:
        return FS_SLOW_COOLING
    if nua0 < nuc0 < num0:
        return FS_FAST_COOLING
    if num0 < nua0 < nuc0:
        return FS_SELF_ABSORBED_SLOW
    raise ValueError(
        "Unsupported forward-shock break ordering at t0. "
        "Supported adiabatic relativistic orderings are "
        f"{FS_SLOW_COOLING}, {FS_FAST_COOLING}, and {FS_SELF_ABSORBED_SLOW}; "
        f"got nu_a={nua0:g}, nu_m={num0:g}, nu_c={nuc0:g}."
    )


def _forward_shock_state_evolve_from_tref(
    tval, t_ref, f_ref, nua_ref, num_ref, nuc_ref, regime, k, p, jet_phase=False
):
    fmax_exp, ba, bm, bc = _forward_shock_temporal_indices(k, p, regime, jet_phase=jet_phase)
    scale = tval / t_ref
    return (
        regime,
        f_ref * scale ** fmax_exp,
        nua_ref * scale ** ba,
        num_ref * scale ** bm,
        nuc_ref * scale ** bc,
    )


def _forward_shock_raw_flux(state, nuval, p):
    regime, fnu_max, nua, num, nuc = state
    return _forward_shock_spectrum(regime, nuval, fnu_max, nua, num, nuc, p)


def _future_crossing_time(t_ref, nu_left, nu_right, b_left, b_right):
    if b_left == b_right or nu_left <= 0 or nu_right <= 0:
        return None
    t_cross = t_ref * (nu_right / nu_left) ** (1 / (b_left - b_right))
    if not np.isfinite(t_cross) or t_cross <= t_ref * (1 + 1e-12):
        return None
    return t_cross


def _forward_shock_next_spectral_transition(t_ref, state, k, p, jet_phase):
    regime, _, nua, num, nuc = state
    _, ba, bm, bc = _forward_shock_temporal_indices(k, p, regime, jet_phase=jet_phase)

    if regime == FS_FAST_COOLING:
        t_cross = _future_crossing_time(t_ref, num, nuc, bm, bc)
        return t_cross, FS_SLOW_COOLING

    if regime == FS_SLOW_COOLING:
        t_cross = _future_crossing_time(t_ref, nua, num, ba, bm)
        return t_cross, FS_SELF_ABSORBED_SLOW

    return None, None


def _forward_shock_next_event(t_ref, state, k, p, jet_break, jet_phase):
    event_time = None
    event_kind = None
    event_regime = None

    if jet_break is not None and not jet_phase and jet_break > t_ref * (1 + 1e-12):
        event_time = jet_break
        event_kind = "jet"

    transition_time, transition_regime = _forward_shock_next_spectral_transition(t_ref, state, k, p, jet_phase)
    if transition_time is not None and (event_time is None or transition_time < event_time):
        event_time = transition_time
        event_kind = "spectral"
        event_regime = transition_regime

    return event_time, event_kind, event_regime


def _forward_shock_initial_state(f0, nua0, num0, nuc0, jet_break, k, t0, p):
    initial_regime = _forward_shock_initial_regime(nua0, num0, nuc0)
    initial_jet_phase = jet_break is not None and jet_break <= t0
    return (initial_regime, f0, nua0, num0, nuc0), initial_jet_phase


def _forward_shock_branch_state(tval, f0, nua0, num0, nuc0, k, t0, p, jet_break=None):
    state, jet_phase = _forward_shock_initial_state(f0, nua0, num0, nuc0, jet_break, k, t0, p)
    t_ref = t0

    if tval <= t_ref:
        return _forward_shock_state_evolve_from_tref(
            tval, t_ref, state[1], state[2], state[3], state[4], state[0], k, p, jet_phase=jet_phase
        )

    while True:
        event_time, event_kind, event_regime = _forward_shock_next_event(t_ref, state, k, p, jet_break, jet_phase)
        if event_time is None or event_time >= tval:
            return _forward_shock_state_evolve_from_tref(
                tval, t_ref, state[1], state[2], state[3], state[4], state[0], k, p, jet_phase=jet_phase
            )

        state = _forward_shock_state_evolve_from_tref(
            event_time, t_ref, state[1], state[2], state[3], state[4], state[0], k, p, jet_phase=jet_phase
        )
        t_ref = event_time
        if event_kind == "jet":
            jet_phase = True
        elif event_kind == "spectral":
            state = (event_regime, state[1], state[2], state[3], state[4])


def _forward_shock_flux_from_state(tval, nuval, t_ref, ref_state, k, p, jet_break, jet_phase, previous_flux_at_ref=None):
    raw_ref = _forward_shock_raw_flux(ref_state, nuval, p)
    if previous_flux_at_ref is None:
        previous_flux_at_ref = raw_ref

    event_time, event_kind, event_regime = _forward_shock_next_event(t_ref, ref_state, k, p, jet_break, jet_phase)
    if event_time is None or event_time >= tval:
        current = _forward_shock_state_evolve_from_tref(
            tval, t_ref, ref_state[1], ref_state[2], ref_state[3], ref_state[4], ref_state[0], k, p, jet_phase=jet_phase
        )
        return previous_flux_at_ref * _forward_shock_raw_flux(current, nuval, p) / raw_ref

    event_state = _forward_shock_state_evolve_from_tref(
        event_time, t_ref, ref_state[1], ref_state[2], ref_state[3], ref_state[4], ref_state[0], k, p, jet_phase=jet_phase
    )
    flux_at_event = previous_flux_at_ref * _forward_shock_raw_flux(event_state, nuval, p) / raw_ref

    if event_kind == "jet":
        next_state = event_state
        next_jet_phase = True
    else:
        next_state = (event_regime, event_state[1], event_state[2], event_state[3], event_state[4])
        next_jet_phase = jet_phase

    return _forward_shock_flux_from_state(
        tval, nuval, event_time, next_state, k, p, jet_break, next_jet_phase, previous_flux_at_ref=flux_at_event
    )


def forward_shock_flux(ivar, f0, nu0_1, nu0_2, nu0_3, k, t0, jet_break=None, p=2.2):
    t, nu = ivar
    initial_state, initial_jet_phase = _forward_shock_initial_state(f0, nu0_1, nu0_2, nu0_3, jet_break, k, t0, p)
    results = []

    for tval, nuval in zip(t, nu):
        result = _forward_shock_flux_from_state(
            tval, nuval, t0, initial_state, k, p, jet_break, initial_jet_phase
        )
        results.append(result)

    return np.array(results)


def _reverse_shock_thick_shell_indices(k, p, regime):
    """Temporal indices for the thick-shell reverse shock in Table 5."""
    fmax = -(47 - 10 * k) / (12 * (4 - k))
    num = -(73 - 14 * k) / (12 * (4 - k))
    nuc = num

    if regime in (RS_SLOW_COOLING, RS_FAST_COOLING):
        nua = -(32 - 7 * k) / (15 * (4 - k))
    elif regime == RS_SELF_ABSORBED_SLOW:
        nua = -(
            p * (73 - 14 * k) + 2 * (67 - 14 * k)
        ) / (12 * (4 - k) * (p + 4))
    else:
        raise ValueError(f"Unsupported reverse-shock spectral regime: {regime}")

    return fmax, nua, num, nuc


def _reverse_shock_initial_regime(nua0, num0, nuc0):
    if nua0 < num0 < nuc0:
        return RS_SLOW_COOLING
    if nua0 < nuc0 < num0:
        return RS_FAST_COOLING
    if num0 < nua0 < nuc0:
        return RS_SELF_ABSORBED_SLOW
    raise ValueError(
        "Unsupported reverse-shock break ordering at t0_rev. "
        "Supported thick-shell orderings are "
        f"{RS_SLOW_COOLING}, {RS_FAST_COOLING}, and {RS_SELF_ABSORBED_SLOW}; "
        f"got nu_a={nua0:g}, nu_m={num0:g}, nu_c={nuc0:g}."
    )


def _crossing_time(t_ref, nu_left, nu_right, b_left, b_right):
    return t_ref * (nu_right / nu_left) ** (1 / (b_left - b_right))


def _reverse_shock_spectrum(regime, nu, fnu_max, nua, num, nuc, p):
    if regime == RS_SLOW_COOLING:
        # nu_a < nu_m < nu_c
        f_at_nua = fnu_max * (nua / num) ** (1 / 3)
        return tsbpl(nu, f_at_nua, nua, num, nuc, 2, 1 / 3, -(p - 1) / 2, -p / 2)

    if regime == RS_FAST_COOLING:
        # nu_a < nu_c < nu_m
        f_at_nua = fnu_max * (nua / nuc) ** (1 / 3)
        return tsbpl(nu, f_at_nua, nua, nuc, num, 2, 1 / 3, -1 / 2, -p / 2)

    if regime == RS_SELF_ABSORBED_SLOW:
        # nu_m < nu_a < nu_c
        f_at_num = fnu_max * (num / nua) ** 3
        return tsbpl(nu, f_at_num, num, nua, nuc, 2, 5 / 2, -(p - 1) / 2, -p / 2)

    raise ValueError(f"Unsupported reverse-shock spectral regime: {regime}")


def _reverse_shock_branch_state(tval, f0, nua0, num0, nuc0, k, t0_rev, p):
    initial_regime = _reverse_shock_initial_regime(nua0, num0, nuc0)

    fmax_i, ba_i, bm_i, bc_i = _reverse_shock_thick_shell_indices(
        k, p, initial_regime
    )

    if initial_regime == RS_SLOW_COOLING:
        t_am = _crossing_time(t0_rev, nua0, num0, ba_i, bm_i)
        if tval <= t_am:
            scale = tval / t0_rev
            return (
                initial_regime,
                f0 * scale ** fmax_i,
                nua0 * scale ** ba_i,
                num0 * scale ** bm_i,
                nuc0 * scale ** bc_i,
                None,
            )

        fmax_t, ba_t, bm_t, bc_t = _reverse_shock_thick_shell_indices(
            k, p, RS_SELF_ABSORBED_SLOW
        )
        scale_cross = t_am / t0_rev
        fmax_cross = f0 * scale_cross ** fmax_i
        nu_cross = nua0 * scale_cross ** ba_i
        nuc_cross = nuc0 * scale_cross ** bc_i
        scale = tval / t_am
        nua = nu_cross * scale ** ba_t
        num = nu_cross * scale ** bm_t
        nuc = nuc_cross * scale ** bc_t
        if not (num < nua < nuc):
            raise ValueError(
                "Unsupported reverse-shock break ordering after nu_a/nu_m "
                f"crossing: nu_m={num:g}, nu_a={nua:g}, nu_c={nuc:g}."
            )
        return (
            RS_SELF_ABSORBED_SLOW,
            fmax_cross * scale ** fmax_t,
            nua,
            num,
            nuc,
            (initial_regime, t_am),
        )

    if initial_regime == RS_FAST_COOLING:
        t_ac = _crossing_time(t0_rev, nua0, nuc0, ba_i, bc_i)
        if tval > t_ac:
            scale = tval / t0_rev
            nua = nua0 * scale ** ba_i
            num = num0 * scale ** bm_i
            nuc = nuc0 * scale ** bc_i
            raise ValueError(
                "Unsupported reverse-shock break ordering after nu_a/nu_c "
                f"crossing: nu_a={nua:g}, nu_c={nuc:g}, nu_m={num:g}."
            )
        scale = tval / t0_rev
        return (
            initial_regime,
            f0 * scale ** fmax_i,
            nua0 * scale ** ba_i,
            num0 * scale ** bm_i,
            nuc0 * scale ** bc_i,
            None,
        )

    t_am = _crossing_time(t0_rev, nua0, num0, ba_i, bm_i)
    if tval >= t_am:
        scale = tval / t0_rev
        return (
            initial_regime,
            f0 * scale ** fmax_i,
            nua0 * scale ** ba_i,
            num0 * scale ** bm_i,
            nuc0 * scale ** bc_i,
            None,
        )

    fmax_s, ba_s, bm_s, bc_s = _reverse_shock_thick_shell_indices(
        k, p, RS_SLOW_COOLING
    )
    scale_cross = t_am / t0_rev
    fmax_cross = f0 * scale_cross ** fmax_i
    nu_cross = nua0 * scale_cross ** ba_i
    nuc_cross = nuc0 * scale_cross ** bc_i
    scale = tval / t_am
    return (
        RS_SLOW_COOLING,
        fmax_cross * scale ** fmax_s,
        nu_cross * scale ** ba_s,
        nu_cross * scale ** bm_s,
        nuc_cross * scale ** bc_s,
        (initial_regime, t_am),
    )


def reverse_shock_break_frequencies(ivar, nua0_rev, num0_rev, nuc0_rev, k, t0_rev, p=2.2):
    """Return thick-shell RS (nu_a, nu_m, nu_c) at the requested times."""
    t, _ = ivar
    t = np.asarray(t)
    nua = []
    num = []
    nuc = []
    for tval in t:
        _, _, nua_t, num_t, nuc_t, _ = _reverse_shock_branch_state(
            tval, 1.0, nua0_rev, num0_rev, nuc0_rev, k, t0_rev, p
        )
        nua.append(nua_t)
        num.append(num_t)
        nuc.append(nuc_t)
    return np.array(nua), np.array(num), np.array(nuc)


# Relativistic thick-shell reverse shock.
def reverse_shock(ivar, f0, nu0_1, nu0_2, nu0_3, k, t0_rev,p=2.2,givenuvals=False):
    t, nu = ivar
    res = []
    nuvals = []
    for tval,nuval in zip(t,nu):
        regime, fnu_max, nua, num, nuc, transition = _reverse_shock_branch_state(
            tval, f0, nu0_1, nu0_2, nu0_3, k, t0_rev, p
        )
        result = _reverse_shock_spectrum(regime, nuval, fnu_max, nua, num, nuc, p)

        if transition is not None:
            previous_regime, t_cross = transition
            _, fmax_old, nua_old, num_old, nuc_old, _ = _reverse_shock_branch_state(
                t_cross, f0, nu0_1, nu0_2, nu0_3, k, t0_rev, p
            )
            old_at_cross = _reverse_shock_spectrum(
                previous_regime, nuval, fmax_old, nua_old, num_old, nuc_old, p
            )
            new_at_cross = _reverse_shock_spectrum(
                regime, nuval, fmax_old, nua_old, num_old, nuc_old, p
            )
            result = old_at_cross * result / new_at_cross

        res.append(result)
        if givenuvals:
            nuvals.append((tval,nua,num,nuc))
    if givenuvals:
        return np.array(res),np.array(nuvals) 
    else:
        return np.array(res)

def forward_shock_break_frequencies(ivar, nua_0, num_0, nuc_0, k, t0, p=2.2, jet_break=None):
    """Return FS (nu_a, nu_m, nu_c) at the requested observer times."""
    t, _ = ivar
    t = np.asarray(t)
    nua = []
    num = []
    nuc = []
    for tval in t:
        _, _, nua_t, num_t, nuc_t = _forward_shock_branch_state(
            tval, 1.0, nua_0, num_0, nuc_0, k, t0, p, jet_break=jet_break
        )
        nua.append(nua_t)
        num.append(num_t)
        nuc.append(nuc_t)

    return np.array(nua), np.array(num), np.array(nuc)


def forward_shock_absorption_tau(
    ivar,
    nua_0,
    num_0,
    nuc_0,
    k,
    t0,
    p=2.2,
    jet_break=None,
):
    """FS optical depth for RS photons from McMahon, Kumar & Piran 2006 Eq. 14."""
    _, nu = ivar
    nu = np.asarray(nu)
    nua, num, nuc = forward_shock_break_frequencies(
        ivar, nua_0, num_0, nuc_0, k, t0, p=p, jet_break=jet_break
    )
    lower_fs_break = np.minimum(nuc, num)
    below_lower_fs_break = nu < lower_fs_break

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tau_low = (nu / nua) ** (-5 / 3)
        tau_high = (nu / nua) ** (-(p + 4) / 2)

    return np.where(below_lower_fs_break, tau_low, tau_high)


def forward_model(ivar, f0, nua_0, num_0, nuc_0, k, t0, p, t_j=None):
    return forward_shock_flux(ivar, f0, nua_0, num_0, nuc_0, k, t0, jet_break=t_j, p=p)


def forward_reverse_model(
    ivar,
    f0, f0_rev,
    nua0_rev, num0_rev, nuc0_rev,
    nua_0, num_0, nuc_0,
    k, t0, t0_rev, p, t_j=None, apply_fs_absorption=False
):
    fwd = forward_shock_flux(ivar, f0, nua_0, num_0, nuc_0, k, t0, jet_break=t_j, p=p)
    rev = reverse_shock(ivar, f0_rev, nua0_rev, num0_rev, nuc0_rev, k, t0_rev, p)
    if not apply_fs_absorption:
        return fwd + rev

    tau_abs_fs = forward_shock_absorption_tau(ivar, nua_0, num_0, nuc_0, k, t0, p=p, jet_break=t_j)
    return fwd + rev * np.exp(-tau_abs_fs)


# Backward-compatible alias for older code/tests that imported this name.
theory_bigsbpl = forward_shock_flux
