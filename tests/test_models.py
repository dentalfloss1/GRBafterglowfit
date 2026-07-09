import unittest

import numpy as np

from grbfit.models import (
    forward_shock_absorption_tau,
    forward_shock_break_frequencies,
    reverse_shock,
    reverse_shock_break_frequencies,
)


class ForwardShockAbsorptionTests(unittest.TestCase):
    def test_absorption_tau_is_continuous_when_lower_break_crosses_band(self):
        nua_0 = 20.82295158421092
        num_0 = 1608.5675270013865
        nuc_0 = 8e8
        k = 2
        t0 = 1.0
        p = 2.2
        freq = 6.55
        t_cross = (num_0 / freq) ** (2 / 3)

        times = t_cross * np.array([0.999, 1.0, 1.001])
        freqs = np.full_like(times, freq)
        tau = forward_shock_absorption_tau(
            (times, freqs),
            nua_0,
            num_0,
            nuc_0,
            k,
            t0,
            p=p,
        )

        self.assertLess(np.max(tau) / np.min(tau), 1.05)

    def test_absorption_tau_matches_asymptotic_branches_far_from_break(self):
        nua_0 = 10.0
        num_0 = 100.0
        nuc_0 = 1e8
        k = 2
        t0 = 1.0
        p = 2.2
        time = 1.0

        lower_break = min(forward_shock_break_frequencies(
            (np.array([time]), np.array([1.0])),
            nua_0,
            num_0,
            nuc_0,
            k,
            t0,
            p=p,
        )[1][0], nuc_0)

        low_freq = lower_break / 100.0
        high_freq = lower_break * 100.0
        freqs = np.array([low_freq, high_freq])
        times = np.full_like(freqs, time)
        tau = forward_shock_absorption_tau(
            (times, freqs),
            nua_0,
            num_0,
            nuc_0,
            k,
            t0,
            p=p,
        )

        low_expected = (low_freq / nua_0) ** (5 / 3)
        tau_break = (lower_break / nua_0) ** (5 / 3)
        high_expected = tau_break * (high_freq / lower_break) ** (-(p + 4) / 2)

        np.testing.assert_allclose(tau[0], low_expected, rtol=1e-8)
        np.testing.assert_allclose(tau[1], high_expected, rtol=1e-8)


class ReverseShockRegimeTests(unittest.TestCase):
    def test_break_frequencies_follow_thick_shell_indices(self):
        k = 2
        p = 2.2
        t0 = 1.0
        nua0 = 10.0
        num0 = 1000.0
        nuc0 = 1e8

        ba_low = -(32 - 7 * k) / (15 * (4 - k))
        bm = -(73 - 14 * k) / (12 * (4 - k))
        ba_thick = -(
            p * (73 - 14 * k) + 2 * (67 - 14 * k)
        ) / (12 * (4 - k) * (p + 4))
        t_cross = t0 * (num0 / nua0) ** (1 / (ba_low - bm))

        early_times = np.array([1.0, 2.0])
        early_freqs = np.ones_like(early_times)
        early_nua, early_num, _ = reverse_shock_break_frequencies(
            (early_times, early_freqs), nua0, num0, nuc0, k, t0, p=p
        )
        measured_ba_low = np.log(early_nua[1] / early_nua[0]) / np.log(2.0)
        measured_bm = np.log(early_num[1] / early_num[0]) / np.log(2.0)

        late_times = t_cross * np.array([2.0, 4.0])
        late_freqs = np.ones_like(late_times)
        late_nua, _, _ = reverse_shock_break_frequencies(
            (late_times, late_freqs), nua0, num0, nuc0, k, t0, p=p
        )
        measured_ba_thick = np.log(late_nua[1] / late_nua[0]) / np.log(2.0)

        self.assertAlmostEqual(measured_ba_low, ba_low)
        self.assertAlmostEqual(measured_bm, bm)
        self.assertAlmostEqual(measured_ba_thick, ba_thick)

    def test_flux_is_continuous_across_nua_num_crossing(self):
        k = 2
        p = 2.2
        t0 = 1.0
        nua0 = 10.0
        num0 = 100.0
        nuc0 = 1e8
        ba_low = -(32 - 7 * k) / (15 * (4 - k))
        bm = -(73 - 14 * k) / (12 * (4 - k))
        t_cross = t0 * (num0 / nua0) ** (1 / (ba_low - bm))

        times = t_cross * np.array([0.999, 1.0, 1.001])
        freqs = np.full_like(times, 8.0)
        flux = reverse_shock((times, freqs), 1e-3, nua0, num0, nuc0, k, t0, p=p)

        self.assertLess(np.max(flux) / np.min(flux), 1.01)

    def test_self_absorbed_reverse_shock_crosses_to_slow_cooling_at_earlier_times(self):
        k = 2
        p = 2.2
        t0 = 0.05
        f0 = 1e-3
        nua0 = 33.0
        num0 = 31.0
        nuc0 = 8e8
        ba_self = -(
            p * (73 - 14 * k) + 2 * (67 - 14 * k)
        ) / (12 * (4 - k) * (p + 4))
        bm = -(73 - 14 * k) / (12 * (4 - k))
        t_cross = t0 * (num0 / nua0) ** (1 / (ba_self - bm))

        times = t_cross * np.array([0.999, 1.0, 1.001])
        freqs = np.full_like(times, 9.0)
        flux, breaks = reverse_shock(
            (times, freqs), f0, nua0, num0, nuc0, k, t0, p=p, givenuvals=True
        )

        self.assertLess(breaks[0, 1], breaks[0, 2])
        self.assertLess(breaks[2, 2], breaks[2, 1])
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux > 0))
        self.assertLess(np.max(flux) / np.min(flux), 1.01)

    def test_supported_regimes_have_expected_asymptotic_slopes(self):
        k = 2
        p = 2.2
        t0 = 1.0
        f0 = 1e-3

        cases = [
            ((10.0, 1e4, 1e8), (1e5, 1e6), -(p - 1) / 2),
            ((10.0, 1e8, 1e4), (1e5, 1e6), -1 / 2),
            ((1e4, 10.0, 1e8), (100.0, 1000.0), 5 / 2),
        ]
        for breaks, freqs, expected_slope in cases:
            nua0, num0, nuc0 = breaks
            freq_arr = np.array(freqs)
            time_arr = np.full_like(freq_arr, t0)
            flux = reverse_shock(
                (time_arr, freq_arr), f0, nua0, num0, nuc0, k, t0, p=p
            )
            measured_slope = np.log(flux[1] / flux[0]) / np.log(freqs[1] / freqs[0])
            self.assertAlmostEqual(measured_slope, expected_slope, places=2)

    def test_unsupported_reverse_shock_ordering_raises_clear_error(self):
        times = np.array([1.0])
        freqs = np.array([10.0])
        with self.assertRaisesRegex(ValueError, "Unsupported reverse-shock break ordering"):
            reverse_shock((times, freqs), 1e-3, 100.0, 1000.0, 10.0, 2, 1.0)


if __name__ == "__main__":
    unittest.main()
