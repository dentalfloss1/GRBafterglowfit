import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grbfit.data import prepare_fit_data
from grbfit.run import plot_best_fit_residuals, plot_non_fitted_residuals


class ResidualPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_excluded_predictions_use_fitted_median_and_measurement_errors(self):
        data = pd.DataFrame({
            "obsdate": [1., 2., 3., 4., 5.],
            "freq": [10., 50., 10., 10., 10.],
            "flux": [200., 300., 50., -1., 100.],
            "err": [30., 30., 3., -1., -1.],
            "rms": [40., 10., 4., 10., 10.],
            "instrument": ["radio"] * 5,
        }, index=[9, 2, 7, 1, 5])
        cfg = {"burst": {"z": None}, "fit": {"min_freq": 50}}
        excluded = prepare_fit_data(data, cfg)[4]
        original = excluded.copy(deep=True)
        samples = np.array([[1.], [5.], [3.]])
        model = Mock(return_value=np.array([100., 100.]) * 1e-6)
        with patch("grbfit.run.make_model", return_value=model), \
                patch("grbfit.run.plt.savefig") as save, \
                patch("grbfit.run.plt.close"):
            plot_non_fitted_residuals(cfg, samples, excluded)
            save.assert_called_once_with("residuals_non_fitted.png", dpi=200)
            points = np.concatenate([artist.get_offsets() for ax in plt.gcf().axes
                                     for artist in ax.collections])
        np.testing.assert_allclose(points, [
            [1., np.log(2.) / 0.25],
            [3., np.log(0.5) / 0.1],
        ])
        model.assert_called_once()
        theta, (times, frequencies) = model.call_args.args
        np.testing.assert_array_equal(theta, [3.])
        np.testing.assert_array_equal(times, [1., 3.])
        np.testing.assert_array_equal(frequencies, [10., 10.])
        pd.testing.assert_frame_equal(excluded, original)

    def test_no_excluded_detections_skips_plot_and_model(self):
        limits = pd.DataFrame({"flux": [-1., 100.], "err": [-1., 0.]})
        for excluded in (None, pd.DataFrame(), limits):
            with self.subTest(excluded=excluded), \
                    patch("grbfit.run.make_model") as model, \
                    patch("grbfit.run.plt.savefig") as save:
                plot_non_fitted_residuals({}, np.array([[1.]]), excluded)
                model.assert_not_called()
                save.assert_not_called()

    def test_fitted_plot_keeps_filename_and_filters_invalid_predictions(self):
        model = Mock(return_value=np.array([1., 0., np.nan]))
        with patch("grbfit.run.make_model", return_value=model), \
                patch("grbfit.run.plt.savefig") as save, \
                patch("grbfit.run.plt.close"):
            plot_best_fit_residuals(
                {}, np.array([[1.]]), (np.array([1., 2., 3.]), np.full(3, 10.)),
                np.ones(3), np.full(3, 0.1), np.array(["radio"] * 3),
            )
            save.assert_called_once_with("residuals.png", dpi=200)
            np.testing.assert_allclose(plt.gcf().axes[0].collections[0].get_offsets(),
                                       [[1., 0.]])


if __name__ == "__main__":
    unittest.main()
