import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grbfit.data import prepare_fit_data
from grbfit.run import plot_posterior_models


class FrequencyCutoffTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "obsdate": [1., 2., 3., 4., 5.],
            "freq": [10., 50., 100., 20., 200.],
            "flux": [100., 200., 300., -1., -1.],
            "err": [10., 20., 30., -1., -1.],
            "rms": [5., 6., 7., 8., 9.],
            "instrument": ["radio", "other", "radio", "radio", "radio"],
        }, index=[9, 2, 7, 1, 5])
        self.cfg = {"burst": {"z": None, "fitstart": 0.01}, "fit": {}}

    def test_cutoff_is_observed_inclusive_and_preserves_reference_rows(self):
        original = self.data.copy(deep=True)
        for z in (None, 4.0):
            for max_rest_freq in (None, 2.47e6):
                with self.subTest(z=z, max_rest_freq=max_rest_freq):
                    self.cfg["burst"]["z"] = z
                    self.cfg["fit"] = {"min_freq": 50}
                    if max_rest_freq is not None:
                        self.cfg["fit"]["max_rest_freq"] = max_rest_freq
                    (t, nu), y, yerr, upper, excluded, instrument = prepare_fit_data(
                        self.data, self.cfg
                    )
                    np.testing.assert_array_equal(nu, [50., 100.])
                    np.testing.assert_array_equal(t, [2., 3.])
                    np.testing.assert_allclose(y, np.array([200., 300.]) * 1e-6)
                    np.testing.assert_allclose(yerr, np.sqrt([436., 949.]) * 1e-6)
                    np.testing.assert_array_equal(instrument, ["other", "radio"])
                    self.assertEqual(upper.index.tolist(), [5])
                    self.assertEqual(excluded.index.tolist(), [9, 1])
                    pd.testing.assert_frame_equal(excluded[original.columns], original.loc[[9, 1]])
        pd.testing.assert_frame_equal(self.data, original)

    def test_omitted_null_and_zero_leave_data_selection_unchanged(self):
        for fit in ({}, {"min_freq": None}, {"min_freq": 0}):
            with self.subTest(fit=fit):
                self.cfg["fit"] = fit
                (_, nu), _, _, upper, excluded, _ = prepare_fit_data(self.data, self.cfg)
                np.testing.assert_array_equal(nu, [10., 50., 100.])
                self.assertEqual(len(upper), 2)
                self.assertTrue(excluded.empty)

    def test_minimum_combines_with_rest_frequency_and_time_cuts(self):
        self.cfg["burst"]["z"] = 1.
        self.cfg["burst"]["fitstart"] = 1.5
        self.cfg["fit"] = {"min_freq": 50, "max_rest_freq": 150, "fit_xrt": True}
        (_, nu), _, _, _, excluded, _ = prepare_fit_data(self.data, self.cfg)
        np.testing.assert_array_equal(nu, [50.])
        self.assertEqual(excluded.index.tolist(), [9, 7, 1, 5])

    def test_invalid_cutoffs_raise_clear_error(self):
        for cutoff in (-1, float("inf"), float("nan"), "50 GHz"):
            with self.subTest(cutoff=cutoff):
                self.cfg["fit"]["min_freq"] = cutoff
                with self.assertRaisesRegex(ValueError, "fit.min_freq.*GHz"):
                    prepare_fit_data(self.data, self.cfg)

    def test_no_remaining_detections_raise_clear_error(self):
        self.cfg["fit"]["min_freq"] = 150
        with self.assertRaisesRegex(ValueError, "No detections remain.*fit.min_freq"):
            prepare_fit_data(self.data, self.cfg)

    def test_excluded_detections_and_upper_limits_are_plotted(self):
        self.cfg["fit"] = {"min_freq": 50, "param_keys": []}
        prepared = prepare_fit_data(self.data, self.cfg)
        model = lambda theta, xdata: np.full_like(xdata[0], 1e-4)
        with patch("grbfit.run.make_model", return_value=model), \
                patch("grbfit.run.plt.savefig"), patch("grbfit.run.plt.close"):
            try:
                plot_posterior_models(self.cfg, np.empty((1, 0)), *prepared, n_draws=1)
                axes = plt.gcf().axes
                labels = [label for ax in axes for label in ax.get_legend_handles_labels()[1]]
                self.assertTrue(any("10" in label and "excluded" in label for label in labels))
                offsets = [tuple(point) for ax in axes for artist in ax.collections
                           for point in artist.get_offsets()]
                self.assertIn((1., 100.), offsets)
                self.assertIn((4., 24.), offsets)
            finally:
                figure = plt.gcf()
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
