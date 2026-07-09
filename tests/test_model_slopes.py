import unittest

import numpy as np

from grbfit.models import dsbpl, reverse_shock, theory_bigsbpl, tsbpl


P = 2.2
F0 = 1e-3
T0 = 1.0


def _log_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.log(y[1] / y[0]) / np.log(x[1] / x[0])


def _temporal_slope(norm_exp, break_exps, spectral_slopes, segment):
    if segment == 0:
        return norm_exp - spectral_slopes[0] * break_exps[0]
    if segment == 1:
        return norm_exp - spectral_slopes[1] * break_exps[0]
    if segment == 2:
        return (
            norm_exp
            + spectral_slopes[1] * (break_exps[1] - break_exps[0])
            - spectral_slopes[2] * break_exps[1]
        )
    if segment == 3:
        return (
            norm_exp
            + spectral_slopes[1] * (break_exps[1] - break_exps[0])
            + spectral_slopes[2] * (break_exps[2] - break_exps[1])
            - spectral_slopes[3] * break_exps[2]
        )
    raise ValueError(f"Unsupported segment: {segment}")


def _sharp_dsbpl(x, amplitude, xb1, alpha1, alpha2, xb2, alpha3):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    first = x < xb1
    middle = (x >= xb1) & (x < xb2)
    last = x >= xb2
    y[first] = amplitude * (x[first] / xb1) ** alpha1
    y[middle] = amplitude * (x[middle] / xb1) ** alpha2
    y[last] = (
        amplitude
        * (xb2 / xb1) ** alpha2
        * (x[last] / xb2) ** alpha3
    )
    return y


def _sharp_tsbpl(x, amplitude, xb1, xb2, xb3, alpha1, alpha2, alpha3, alpha4):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    first = x < xb1
    second = (x >= xb1) & (x < xb2)
    third = (x >= xb2) & (x < xb3)
    fourth = x >= xb3
    y[first] = amplitude * (x[first] / xb1) ** alpha1
    y[second] = amplitude * (x[second] / xb1) ** alpha2
    y[third] = (
        amplitude
        * (xb2 / xb1) ** alpha2
        * (x[third] / xb2) ** alpha3
    )
    y[fourth] = (
        amplitude
        * (xb2 / xb1) ** alpha2
        * (xb3 / xb2) ** alpha3
        * (x[fourth] / xb3) ** alpha4
    )
    return y


def _forward_flux(times, freqs, k, nua0=1e2, num0=1e8, nuc0=1e15):
    return theory_bigsbpl(
        (np.asarray(times, dtype=float), np.asarray(freqs, dtype=float)),
        F0,
        nua0,
        num0,
        nuc0,
        k,
        T0,
        p=P,
    )


def _forward_early_indices(k):
    a_fmax = -k / (2 * (4 - k))
    b_a = -3 * k / (5 * (4 - k))
    b_m = -3 / 2
    b_c = -(4 - 3 * k) / (2 * (4 - k))
    norm_exp = a_fmax + (b_a - b_m) / 3
    slopes = [2, 1 / 3, -(P - 1) / 2, -P / 2]
    return norm_exp, [b_a, b_m, b_c], slopes


def _forward_late_indices(k):
    a_fmax = -k / (2 * (4 - k))
    b_m = -3 / 2
    b_a = -(12 * P + 8 - 3 * P * k + 2 * k) / (2 * (4 - k) * (P + 4))
    b_c = -(4 - 3 * k) / (2 * (4 - k))
    norm_exp = a_fmax + 3 * (b_m - b_a)
    slopes = [2, 5 / 2, -(P - 1) / 2, -P / 2]
    return norm_exp, [b_m, b_a, b_c], slopes


def _forward_transition_time(k, nua0=1e2, num0=1e8):
    b_a = -3 * k / (5 * (4 - k))
    b_m = -3 / 2
    return T0 * (num0 / nua0) ** (1 / (b_a - b_m))


def _forward_late_code_breaks(time, k, nua0=1e2, num0=1e8, nuc0=1e15):
    b_a_early = -3 * k / (5 * (4 - k))
    b_m = -3 / 2
    b_c = -(4 - 3 * k) / (2 * (4 - k))
    b_a_late = -(12 * P + 8 - 3 * P * k + 2 * k) / (2 * (4 - k) * (P + 4))
    t_cross = _forward_transition_time(k, nua0=nua0, num0=num0)
    nu_cross = nua0 * (t_cross / T0) ** b_a_early
    nuc_cross = nuc0 * (t_cross / T0) ** b_c
    return [
        nu_cross * (time / T0) ** b_m,
        nu_cross * (time / T0) ** b_a_late,
        nuc_cross * (time / T0) ** b_c,
    ]


def _reverse_flux(times, freqs, nua0, num0, nuc0, k=2):
    return reverse_shock(
        (np.asarray(times, dtype=float), np.asarray(freqs, dtype=float)),
        F0,
        nua0,
        num0,
        nuc0,
        k,
        T0,
        p=P,
    )


def _reverse_indices(k, regime):
    fmax = -(47 - 10 * k) / (12 * (4 - k))
    b_m = -(73 - 14 * k) / (12 * (4 - k))
    b_c = b_m
    if regime in ("slow", "fast"):
        b_a = -(32 - 7 * k) / (15 * (4 - k))
    else:
        b_a = -(
            P * (73 - 14 * k) + 2 * (67 - 14 * k)
        ) / (12 * (4 - k) * (P + 4))

    if regime == "slow":
        norm_exp = fmax + (b_a - b_m) / 3
        return norm_exp, [b_a, b_m, b_c], [2, 1 / 3, -(P - 1) / 2, -P / 2]
    if regime == "fast":
        norm_exp = fmax + (b_a - b_c) / 3
        return norm_exp, [b_a, b_c, b_m], [2, 1 / 3, -1 / 2, -P / 2]
    if regime == "self_absorbed":
        norm_exp = fmax + 3 * (b_m - b_a)
        return norm_exp, [b_m, b_a, b_c], [2, 5 / 2, -(P - 1) / 2, -P / 2]
    raise ValueError(f"Unsupported regime: {regime}")


class SmoothBreakPrimitiveTests(unittest.TestCase):
    def test_dsbpl_sharp_limit_matches_piecewise_broken_power_law(self):
        x = np.array([0.2, 5.0, 500.0])
        params = (3.0, 1.0, 2.0, 1 / 3, 100.0, -0.6)

        smooth = dsbpl(x, *params, s=0.05)
        sharp = _sharp_dsbpl(x, *params)

        np.testing.assert_allclose(smooth, sharp, rtol=0.02)

    def test_tsbpl_sharp_limit_matches_piecewise_broken_power_law(self):
        x = np.array([0.2, 5.0, 500.0, 50000.0])
        params = (3.0, 1.0, 100.0, 10000.0, 2.0, 1 / 3, -0.6, -1.1)

        smooth = tsbpl(x, *params, s=0.05)
        sharp = _sharp_tsbpl(x, *params)

        np.testing.assert_allclose(smooth, sharp, rtol=0.02)


class ForwardShockSlopeTests(unittest.TestCase):
    def test_early_branch_spectral_slopes_match_expected_segments(self):
        cases = [
            ([1.0, 10.0], 2),
            ([1e4, 1e5], 1 / 3),
            ([1e10, 1e11], -(P - 1) / 2),
            ([1e17, 1e18], -P / 2),
        ]
        for k in (0, 2):
            for freqs, expected in cases:
                flux = _forward_flux([T0, T0], freqs, k)
                self.assertAlmostEqual(
                    _log_slope(freqs, flux), expected, delta=0.03
                )

    def test_early_branch_temporal_slopes_match_expected_segments(self):
        freqs = [1.0, 1e5, 1e11, 1e17]
        times = [T0, 2 * T0]
        for k in (0, 2):
            norm_exp, break_exps, slopes = _forward_early_indices(k)
            for segment, freq in enumerate(freqs):
                flux = _forward_flux(times, [freq, freq], k)
                expected = _temporal_slope(norm_exp, break_exps, slopes, segment)
                self.assertAlmostEqual(
                    _log_slope(times, flux), expected, delta=0.05
                )

    def test_late_branch_temporal_slopes_match_expected_segments(self):
        for k in (0, 2):
            t_start = 10 * _forward_transition_time(k)
            times = [t_start, 2 * t_start]
            breaks = _forward_late_code_breaks(t_start, k)
            freqs = [
                breaks[0] / 1e3,
                np.sqrt(breaks[0] * breaks[1]),
                np.sqrt(breaks[1] * breaks[2]),
                breaks[2] * 1e3,
            ]
            norm_exp, break_exps, slopes = _forward_late_indices(k)

            for segment, freq in enumerate(freqs):
                temporal_flux = _forward_flux(times, [freq, freq], k)
                expected_temporal = _temporal_slope(
                    norm_exp, break_exps, slopes, segment
                )
                self.assertAlmostEqual(
                    _log_slope(times, temporal_flux),
                    expected_temporal,
                    delta=0.05,
                )

    def test_forward_shock_has_no_hard_spectral_or_temporal_breaks(self):
        for k in (0, 2):
            for break_freq in [1e2, 1e8, 1e15]:
                freqs = break_freq * np.array([0.999, 1.0, 1.001])
                flux = _forward_flux(np.full_like(freqs, T0), freqs, k)
                self.assertTrue(np.all(np.isfinite(flux)))
                self.assertTrue(np.all(flux > 0))
                self.assertLess(np.max(flux) / np.min(flux), 1.02)

            t_cross = _forward_transition_time(k)
            times = t_cross * np.array([0.999, 1.0, 1.001])
            flux = _forward_flux(times, np.full_like(times, 1.0), k)
            self.assertTrue(np.all(np.isfinite(flux)))
            self.assertTrue(np.all(flux > 0))
            self.assertLess(np.max(flux) / np.min(flux), 1.02)


class ReverseShockSlopeTests(unittest.TestCase):
    def test_supported_regime_spectral_slopes_match_expected_segments(self):
        cases = [
            ((1e2, 1e8, 1e15), "slow", [1.0, 1e5, 1e11, 1e17]),
            ((1e2, 1e15, 1e8), "fast", [1.0, 1e5, 1e11, 1e17]),
            ((1e8, 1e2, 1e15), "self_absorbed", [1.0, 1e5, 1e11, 1e17]),
        ]
        for breaks, regime, freqs in cases:
            _, _, slopes = _reverse_indices(2, regime)
            for segment, expected in enumerate(slopes):
                freq_pair = [freqs[segment], freqs[segment] * 10]
                flux = _reverse_flux([T0, T0], freq_pair, *breaks)
                self.assertAlmostEqual(
                    _log_slope(freq_pair, flux), expected, delta=0.03
                )

    def test_supported_regime_temporal_slopes_match_expected_segments(self):
        cases = [
            ((1e2, 1e8, 1e15), "slow", [1.0, 1e5, 1e11, 1e17]),
            ((1e2, 1e15, 1e8), "fast", [1.0, 1e5, 1e11, 1e17]),
            ((1e8, 1e2, 1e15), "self_absorbed", [1.0, 1e5, 1e11, 1e17]),
        ]
        times = [T0, 2 * T0]
        for breaks, regime, freqs in cases:
            norm_exp, break_exps, slopes = _reverse_indices(2, regime)
            for segment, freq in enumerate(freqs):
                flux = _reverse_flux(times, [freq, freq], *breaks)
                expected = _temporal_slope(norm_exp, break_exps, slopes, segment)
                self.assertAlmostEqual(
                    _log_slope(times, flux), expected, delta=0.05
                )

    def test_reverse_shock_has_no_hard_spectral_or_temporal_breaks(self):
        regime_breaks = [
            (1e2, 1e8, 1e15),
            (1e2, 1e15, 1e8),
            (1e8, 1e2, 1e15),
        ]
        for breaks in regime_breaks:
            for break_freq in sorted(breaks):
                freqs = break_freq * np.array([0.999, 1.0, 1.001])
                flux = _reverse_flux(np.full_like(freqs, T0), freqs, *breaks)
                self.assertTrue(np.all(np.isfinite(flux)))
                self.assertTrue(np.all(flux > 0))
                self.assertLess(np.max(flux) / np.min(flux), 1.02)

        nua0, num0, nuc0 = 1e2, 1e4, 1e15
        b_a = -(32 - 7 * 2) / (15 * (4 - 2))
        b_m = -(73 - 14 * 2) / (12 * (4 - 2))
        t_cross = T0 * (num0 / nua0) ** (1 / (b_a - b_m))
        times = t_cross * np.array([0.999, 1.0, 1.001])
        flux = _reverse_flux(times, np.full_like(times, 1.0), nua0, num0, nuc0)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux > 0))
        self.assertLess(np.max(flux) / np.min(flux), 1.02)


if __name__ == "__main__":
    unittest.main()
