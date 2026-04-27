import numpy as np


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


def theory_bigsbpl(ivar, f0, nu0_1, nu0_2, nu0_3, k, t0, p=2.2):
    t, nu = ivar

    a1 = -k / (2 * (4 - k))
    b1 = -3 * k / (5 * (4 - k))
    b2 = -3 / 2
    b3 = -(4 - 3 * k) / (2 * (4 - k))

    result = []

    for tval, nuval in zip(t, nu):
        nua = nu0_1 * (tval / t0) ** b1
        num = nu0_2 * (tval / t0) ** b2
        nuc = nu0_3 * (tval / t0) ** b3

        fnu_m = f0 * (tval / t0) ** a1
        fpk = fnu_m * (nua / num) ** (1 / 3)

        val = tsbpl(nuval, fpk, nua, num, nuc, 2, 1/3, -(p-1)/2, -p/2)
        result.append(val)

    return np.array(result)


def reverse_shock(ivar, f0, nu0_1, k, t0, p=2.2):
    t, nu = ivar

    a1 = -(47 - 10 * k) / (12 * (4 - k))
    b1 = -(32 - 7 * k) / (15 * (4 - k))

    res = []

    for tval, nuval in zip(t, nu):
        fnu_m = f0 * (tval / t0) ** a1
        nua = nu0_1 * (tval / t0) ** b1
        res.append(fnu_m * (nuval / nua) ** 2)

    return np.array(res)


def forward_model(ivar, f0, nua_0, num_0, nuc_0, k, t0, p):
    return theory_bigsbpl(ivar, f0, nua_0, num_0, nuc_0, k, t0, p)


def forward_reverse_model(
    ivar,
    f0, f0_rev,
    nua0_rev,
    nua_0, num_0, nuc_0,
    k, t0, p
):
    fwd = theory_bigsbpl(ivar, f0, nua_0, num_0, nuc_0, k, t0, p)
    rev = reverse_shock(ivar, f0_rev, nua0_rev, k, t0, p)
    return fwd + rev
