import numpy as np

FS_ABSORPTION_SMOOTH_WIDTH = 0.2

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


def _forward_shock_slow_to_self_absorbed_crossing(nua0, num0, k, t0):
    _, ba, bm, _ = _forward_shock_relativistic_indices(k, 2.2, FS_SLOW_COOLING)
    return t0 * (num0 / nua0) ** (1 / (ba - bm))


def _legacy_forward_shock_late_breaks(tval, nua0, num0, nuc0, k, t0, p):
    """Current late-branch break normalization, retained for the structure-only refactor."""
    _, ba_early, bm, bc = _forward_shock_relativistic_indices(k, p, FS_SLOW_COOLING)
    _, ba_late, _, _ = _forward_shock_relativistic_indices(k, p, FS_SELF_ABSORBED_SLOW)
    t_cross = t0 * (num0 / nua0) ** (1 / (ba_early - bm))
    nu_cross = nua0 * (t_cross / t0) ** ba_early
    nuc_cross = nuc0 * (t_cross / t0) ** bc
    return (
        nu_cross * (tval / t0) ** ba_late,
        nu_cross * (tval / t0) ** bm,
        nuc_cross * (tval / t0) ** bc,
    )


def _forward_shock_slow_branch_state(tval, f0, nua0, num0, nuc0, k, t0, p):
    fmax_exp, ba, bm, bc = _forward_shock_relativistic_indices(k, p, FS_SLOW_COOLING)
    scale = tval / t0
    return (
        FS_SLOW_COOLING,
        f0 * scale ** fmax_exp,
        nua0 * scale ** ba,
        num0 * scale ** bm,
        nuc0 * scale ** bc,
    )


def _forward_shock_legacy_self_absorbed_state(tval, f0, nua0, num0, nuc0, k, t0, p):
    fmax_exp, _, _, _ = _forward_shock_relativistic_indices(k, p, FS_SLOW_COOLING)
    nua, num, nuc = _legacy_forward_shock_late_breaks(tval, nua0, num0, nuc0, k, t0, p)
    return (
        FS_SELF_ABSORBED_SLOW,
        f0 * (tval / t0) ** fmax_exp,
        nua,
        num,
        nuc,
    )


def forward_shock_flux(ivar, f0, nu0_1, nu0_2, nu0_3, k, t0, jet_break=None, p=2.2):
    if jet_break is not None:
        return _legacy_forward_shock_flux(
            ivar, f0, nu0_1, nu0_2, nu0_3, k, t0, jet_break=jet_break, p=p
        )

    t, nu = ivar
    _, ba, bm, _ = _forward_shock_relativistic_indices(k, p, FS_SLOW_COOLING)
    t_cross = t0 * (nu0_2 / nu0_1) ** (1 / (ba - bm))
    results = []

    for tval, nuval in zip(t, nu):
        if tval <= t_cross:
            regime, fnu_max, nua, num, nuc = _forward_shock_slow_branch_state(
                tval, f0, nu0_1, nu0_2, nu0_3, k, t0, p
            )
            result = _forward_shock_spectrum(regime, nuval, fnu_max, nua, num, nuc, p)
        else:
            regime, fnu_max, nua, num, nuc = _forward_shock_legacy_self_absorbed_state(
                tval, f0, nu0_1, nu0_2, nu0_3, k, t0, p
            )
            raw_late = _forward_shock_spectrum(
                regime, nuval, fnu_max, nua, num, nuc, p
            )
            _, fmax_cross, nua_cross, num_cross, nuc_cross = _forward_shock_slow_branch_state(
                t_cross, f0, nu0_1, nu0_2, nu0_3, k, t0, p
            )
            old_at_cross = _forward_shock_spectrum(
                FS_SLOW_COOLING,
                nuval,
                fmax_cross,
                nua_cross,
                num_cross,
                nuc_cross,
                p,
            )
            _, fmax_late_cross, nua_late_cross, num_late_cross, nuc_late_cross = (
                _forward_shock_legacy_self_absorbed_state(
                    t_cross, f0, nu0_1, nu0_2, nu0_3, k, t0, p
                )
            )
            raw_late_at_cross = _forward_shock_spectrum(
                FS_SELF_ABSORBED_SLOW,
                nuval,
                fmax_late_cross,
                nua_late_cross,
                num_late_cross,
                nuc_late_cross,
                p,
            )
            result = old_at_cross * raw_late / raw_late_at_cross
        results.append(result)

    return np.array(results)


def _legacy_forward_shock_flux(ivar, f0, nu0_1, nu0_2,nu0_3, k, t0,jet_break=None, p=2.2):
    
    d=0.1
    s = 10
    a1 = -k/(2*(4-k))
    aj = -1
    b1 = -3*k/(5*(4-k))
    b1j = -1/5
    b2 = -3/2
    b2j = -2
    b3 = -(4-3*k)/(2*(4-k))
    b3j = 0
    t, nu = ivar
    y1 = []
    y2 = []
    y3 = []
    t_break = np.amax(t)
    ## Use an analytic solution here instead of this solution
    t_break = t0*(nu0_2/nu0_1)**(1/(b1-b2))
    nu_trans = nu0_1*(t_break/t0)**b1
    nuc_trans = nu0_3*(t_break/t0)**b3
    if jet_break is not None:
        nua_atjetbreak = nu0_1*(jet_break/t0)**b1
        num_atjetbreak = nu0_2*(jet_break/t0)**b2
        nuc_atjetbreak = nu0_3*(jet_break/t0)**b3
        if nua_atjetbreak<num_atjetbreak:
            t_break = jet_break*(num_atjetbreak/nua_atjetbreak)**(1/(b1j-b2j))
            nu_trans = nua_atjetbreak*(t_break/jet_break)**b1j
            nuc_trans = num_atjetbreak*(t_break/jet_break)**b3j
    for tval,nuval in zip(t,nu):
        b1_1 = -3*k/(5*(4-k))
        b2_1 = -3/2
        b3_1 = -(4-3*k)/(2*(4-k))
        nua_1 = nu0_1*(tval/t0)**b1_1
        num_1 = nu0_2*(tval/t0)**b2_1
        nuc_1 = nu0_3*(tval/t0)**b3_1
        ##### Used to be an if here ### 
        c1_1 = 2
        c2_1 = 1/3
        c3_1 = -(p-1)/2
        c4_1 = -p/2
        fnu_m1 = f0*(tval/t0)**a1
        fpk_1 = fnu_m1*(nua_1/num_1)**(1/3)

        #### ######################
        a1_2 = -k/(2*(4-k))
        b1_2 = -3/2
        b2_2 = -(12*p+8-3*p*k+2*k)/(2*(4-k)*(p+4))
        b3_2 = -(4-3*k)/(2*(4-k))
        c1_2 = 2
        c2_2 = 5/2
        c3_2 = -(p-1)/2
        c4_2 = -p/2
        # At t_break num==nua, determines the normalization of these values.
        num_2 = nu_trans*(tval/t0)**b1_2
        nua_2 = nu_trans*(tval/t0)**b2_2
        nuc_2 = nuc_trans*(tval/t0)**b3_2
        num_break2 = nu_trans*(t_break/t0)**b1_2
        nua_break2 = nu_trans*(t_break/t0)**b2_2
        nuc_break2 = nuc_trans*(t_break/t0)**b3_2
        
        res1 = tsbpl(nuval,fpk_1,nua_1,num_1,nuc_1,c1_1,c2_1,c3_1,c4_1)

        num_1_tbreak = nu0_1*(t_break/t0)**b1_1
        nua_1_tbreak = nu0_2*(t_break/t0)**(b2_1)
        nuc_1_tbreak = nu0_3*(t_break/t0)**(b3_1)
        fnu_m1 = f0*(t_break/t0)**a1
        F_bk1 = tsbpl(nuval, fnu_m1*(nua_1_tbreak/num_1_tbreak)**(1/3), nua_1_tbreak, num_1_tbreak, nuc_1_tbreak, c1_1,c2_1,c3_1,c4_1)
        # F_bk1 = dsbpl(nuval,fpk_1,num_1,c1_1,c2_1,nua_1,c3_1,s)
        if jet_break is not None:
            ######## jet scalings ########
            if t_break < jet_break:

                
                fnu_m_2 = f0*(tval/t0)**a1_2
                fpk_2 = fnu_m_2*(num_2/nua_2)**(3)
                fpk2_break = f0*(t_break/t0)**a1_2*(num_break2/nua_break2)**(3)
                F_bk2 = tsbpl(nuval,fpk2_break, num_break2, nua_break2, nuc_break2, c1_2, c2_2, c3_2, c4_2)
                res2 = F_bk1*tsbpl(nuval,fpk_2,num_2,nua_2,nuc_2,c1_2,c2_2,c3_2,c4_2)/F_bk2

                
                b1_j = -2
                b2_j = -(2*(p+1))/(p+4)
                b3_j = 0
                num_2_atjetbreak = nu_trans*(jet_break/t0)**b1_2
                nua_2_atjetbreak = nu_trans*(jet_break/t0)**b2_2
                nuc_2_atjetbreak = nuc_trans*(jet_break/t0)**b3_2
                
                num_j = num_2_atjetbreak*(tval/jet_break)**b1_j
                nua_j = nua_2_atjetbreak*(tval/jet_break)**b2_j
                nuc_j = nuc_2_atjetbreak*(tval/jet_break)**b3_j
                
                fpk_j = (f0*(tval/t0)**aj)*(num_j/nua_j)**(3)
                
                
                fpkj_break = (f0*(jet_break/t0)**aj)*(num_2_atjetbreak/nua_2_atjetbreak)**(3)
                F_bkj = tsbpl(nuval,fpkj_break, num_2_atjetbreak, nua_2_atjetbreak, nuc_2_atjetbreak, c1_2, c2_2, c3_2, c4_2)

                fpk_2_jetbreak = (f0*(jet_break/t0)**a1_2)*((num_2_atjetbreak/nua_2_atjetbreak)**(3))
                F_bk2j = F_bk1*tsbpl(nuval,fpk_2_jetbreak,num_2_atjetbreak,nua_2_atjetbreak,nuc_2_atjetbreak,c1_2,c2_2,c3_2,c4_2)/F_bk2

                
                resj = F_bk2j*tsbpl(nuval,fpk_j,num_j,nua_j,nuc_j,c1_2,c2_2,c3_2,c4_2)/F_bkj
                # if np.abs(tval-jet_break)<1:
                    # print(fpk_j, fpkj_break)
            else:
               
                b1_j = -1/5
                b2_j = -2
                b3_j = 0

                a1_2 = -1
                b1_2 = -2
                b2_2 = -(2*(p+1))/(p+4)
                b3_2 = 0
 

                
                nua_j = nua_atjetbreak*(tval/jet_break)**b1_j
                num_j = num_atjetbreak*(tval/jet_break)**b2_j
                nuc_j = nuc_atjetbreak*(tval/jet_break)**b3_j
                fpk_j = (f0*(tval/t0)**aj)*(nua_j/num_j)**(1/3)
                Fpk_j_break = (f0*(jet_break/t0)**aj)*(nua_atjetbreak/num_atjetbreak)**(1/3)
                Fpk1_j_break = (f0*(jet_break/t0)**a1)*(nua_atjetbreak/num_atjetbreak)**(1/3)


                # fpkj_break = f0*(nua_atjetbreak/numbreak_j)**(1/3)**(jet_break/t0)**aj
                F_bk1j = tsbpl(nuval,Fpk1_j_break,nua_atjetbreak,num_atjetbreak,nuc_atjetbreak,c1_1,c2_1,c3_1,c4_1)
                F_bkj = tsbpl(nuval,Fpk_j_break, nua_atjetbreak, num_atjetbreak, nuc_atjetbreak, c1_1, c2_1, c3_1, c4_1)
                resj = F_bk1j*tsbpl(nuval,fpk_j,nua_j,num_j,nuc_j,c1_1,c2_1,c3_1,c4_1)/F_bkj
                # resj = tsbpl(nuval,fpk_j,nua_j,num_j,nuc_j,c1_1,c2_1,c3_1,c4_1)

                # fpk_j_trans = f0*(nua_j/num_j)**(1/3)*(t_break/t0)**aj
                # F_bkj_trans = tsbpl(nuval,fpk2_break, nua_atjetbreak, num_atjetbreak, nuc_atjetbreak, c1_1, c2_1, c3_1, c4_1)
                # fpk2_break = Fpk1_j_break*(t_break/t0)**a1_2*(num_break2/nua_break2)**(3)
                
          
                b1_2j = -2
                b2_2j = -(2*(p+1))/(p+4)
                b3_2j = 0

                num_2 = nu_trans*(tval/t_break)**b1_2j
                nua_2 = nu_trans*(tval/t_break)**b2_2j
                nuc_2 = nuc_trans*(tval/t_break)**b3_2j

                num_j_tbreak = nua_atjetbreak*(t_break/jet_break)**b1_j
                nua_j_tbreak = num_atjetbreak*(t_break/jet_break)**b2_j
                nuc_j_tbreak = nuc_atjetbreak*(t_break/jet_break)**b3_j
                
                fpk2_break = (f0*(t_break/t0)**aj) # num/nua factor is one by definition
                F_bkj_tbreak = F_bk1j*tsbpl(nuval,fpk2_break, num_j_tbreak, nua_j_tbreak, nuc_j_tbreak, c1_1, c2_1, c3_1, c4_1)/F_bkj

                Fbk2_tbreak = tsbpl(nuval,fpk2_break, nu_trans, nu_trans, nuc_trans, c1_2, c2_2, c3_2, c4_2)
                fnu_m_2 = f0*(tval/t0)**a1_2
                fpk_2 = fnu_m_2*(num_2/nua_2)**(3)
                res2 = F_bkj_tbreak*tsbpl(nuval,fpk_2,num_2,nua_2,nuc_2,c1_2,c2_2,c3_2,c4_2)/Fbk2_tbreak
            
                # if np.abs(tval-t_break)<10:
                # if num_j < nua_j:
                #     print(tval,t_break,nua_j,num_j)
                # print("A------>",nua_j,num_j)
                # fpk2_break = f0*(t_break/t0)**a1_2*(num_break2/nua_break2)**(3)
                # F_bk2 = tsbpl(nuval,fpk2_break, num_break2, nua_break2, nuc_break2, c1_2, c2_2, c3_2, c4_2)
                # res2 = F_bk2*tsbpl(nuval,fpk_2,num_2,nua_2,nuc_2,c1_2,c2_2,c3_2,c4_2)/F_bkj
                # print(F_bk1,F_bk2,F_bkj,F_bk2/F_bkj)
                # res2 = tsbpl(nuval,fpk_2,num_2,nua_2,nuc_2,c1_2,c2_2,c3_2,c4_2)

        else:
            fnu_m_2 = f0*(tval/t0)**a1_2
            fpk_2 = fnu_m_2*(num_2/nua_2)**(3)
            fpk2_break = f0*(t_break/t0)**a1_2*(num_break2/nua_break2)**(3)
            F_bk2 = tsbpl(nuval,fpk2_break, num_break2, nua_break2, nuc_break2, c1_2, c2_2, c3_2, c4_2)
            res2 = F_bk1*tsbpl(nuval,fpk_2,num_2,nua_2,nuc_2,c1_2,c2_2,c3_2,c4_2)/F_bk2
            resj = None
        y1.append(res1)
        y2.append(res2)
        y3.append(resj)
        # y2.append(res2)
    # result = np.where( t<=t_break,np.array(y1),np.array(y2))
    if jet_break is None:
        result = np.where( t<=t_break,np.array(y1),np.array(y2))
    else:
        if t_break < jet_break:
            result = np.where( t<=t_break,np.array(y1),np.array(y2))
            result = np.where(t>jet_break,np.array(y3),result)
        else:
            result = np.where(t<=jet_break,np.array(y1),np.array(y3))
            result = np.where(t>t_break,np.array(y2),result)
    return result



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

    scale = tval / t0_rev
    nua = nua0 * scale ** ba_i
    num = num0 * scale ** bm_i
    nuc = nuc0 * scale ** bc_i
    if not (num < nua < nuc):
        raise ValueError(
            "Unsupported reverse-shock break ordering for self-absorbed "
            f"slow-cooling branch: nu_m={num:g}, nu_a={nua:g}, nu_c={nuc:g}."
        )
    return (
        initial_regime,
        f0 * scale ** fmax_i,
        nua,
        num,
        nuc,
        None,
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

def forward_shock_break_frequencies(ivar, nua_0, num_0, nuc_0, k, t0, p=2.2):
    """Return FS (nu_a, nu_m, nu_c) at the requested observer times."""
    t, _ = ivar
    t = np.asarray(t)

    b_nua_early = -3 * k / (5 * (4 - k))
    b_num = -3 / 2
    b_nuc = -(4 - 3 * k) / (2 * (4 - k))
    b_nua_late = -(12 * p + 8 - 3 * p * k + 2 * k) / (2 * (4 - k) * (p + 4))

    t_cross = t0 * (num_0 / nua_0) ** (1 / (b_nua_early - b_num))
    nu_cross = nua_0 * (t_cross / t0) ** b_nua_early

    nua_early = nua_0 * (t / t0) ** b_nua_early
    num = num_0 * (t / t0) ** b_num
    nuc = nuc_0 * (t / t0) ** b_nuc
    nua_late = nu_cross * (t / t_cross) ** b_nua_late
    nua = np.where(t <= t_cross, nua_early, nua_late)

    return nua, num, nuc


def forward_shock_absorption_tau(
    ivar,
    nua_0,
    num_0,
    nuc_0,
    k,
    t0,
    p=2.2,
    smooth_width=FS_ABSORPTION_SMOOTH_WIDTH,
):
    """FS optical depth for RS photons from McMahon, Kumar & Piran 2006 Eq. 14."""
    _, nu = ivar
    nu = np.asarray(nu)
    nua, num, nuc = forward_shock_break_frequencies(
        ivar, nua_0, num_0, nuc_0, k, t0, p=p
    )
    lower_fs_break = np.minimum(nuc, num)
    low_slope = 5 / 3
    high_slope = -(p + 4) / 2

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_tau_low = low_slope * np.log(nu / nua)
        log_tau_break = low_slope * np.log(lower_fs_break / nua)
        log_tau_high = log_tau_break + high_slope * np.log(nu / lower_fs_break)

        if smooth_width <= 0:
            below_lower_fs_break = nu < lower_fs_break
            return np.where(
                below_lower_fs_break,
                np.exp(log_tau_low),
                np.exp(log_tau_high),
            )

        log_break_ratio = np.log(nu / lower_fs_break)
        high_weight = 0.5 * (1.0 + np.tanh(log_break_ratio / smooth_width))
        log_tau = (1.0 - high_weight) * log_tau_low + high_weight * log_tau_high
        return np.exp(log_tau)


def forward_model(ivar, f0, nua_0, num_0, nuc_0, k, t0, p, t_j=None):
    return forward_shock_flux(ivar, f0, nua_0, num_0, nuc_0, k, t0, jet_break=t_j, p=p)


def forward_reverse_model(
    ivar,
    f0, f0_rev,
    nua0_rev, num0_rev, nuc0_rev,
    nua_0, num_0, nuc_0,
    k, t0, t0_rev, p, t_j=None
):
    fwd = forward_shock_flux(ivar, f0, nua_0, num_0, nuc_0, k, t0, jet_break=t_j, p=p)
    rev = reverse_shock(ivar, f0_rev, nua0_rev, num0_rev, nuc0_rev, k, t0_rev, p)
    tau_abs_fs = forward_shock_absorption_tau(ivar, nua_0, num_0, nuc_0, k, t0, p=p)
    return fwd + rev * np.exp(-tau_abs_fs)


# Backward-compatible alias for older code/tests that imported this name.
theory_bigsbpl = forward_shock_flux
