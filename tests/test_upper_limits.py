import copy
import unittest
from importlib import resources
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr

from grbfit.data import prepare_fit_data, prepare_upper_limit_data
from grbfit.fit import (
    build_param_vector, log_likelihood, log_probability, make_model,
    run_mcmc, theta_to_physical, upper_limit_log_likelihood,
)
from grbfit.run import calculate_goodness_of_fit, format_goodness_of_fit_report, normalize_config


def constant_model(theta, xdata):
    return np.full_like(xdata[0], theta[0], dtype=float)


def config():
    return yaml.safe_load(resources.files("grbfit").joinpath("config_template.yaml").read_text())


class UpperLimitTests(unittest.TestCase):
    def setUp(self):
        self.x = (np.array([1.]), np.array([10.]))
        self.upper = (self.x, np.array([3.]), np.array([1.]))
        self.frame = pd.DataFrame({
            "obsdate": [1., 2., 3., 4.], "freq": [10., 10., 5., 1000.],
            "flux": [100., -1., -1., -1.], "err": [10., -1., -1., -1.],
            "rms": [5., -2., 4., 5.], "instrument": ["radio"] * 4,
        }, index=[4, 7, 10, 12])

    def test_cdf_values_and_extreme_tail(self):
        values = [upper_limit_log_likelihood([flux], constant_model, self.upper)
                  for flux in (0.01, 3., 10., 1003.)]
        self.assertAlmostEqual(values[0], log_ndtr(2.99))
        self.assertAlmostEqual(values[1], np.log(0.5))
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_invalid_models_and_noise_are_rejected(self):
        for flux in (0., -1., np.nan, np.inf):
            self.assertEqual(upper_limit_log_likelihood([flux], constant_model, self.upper), -np.inf)
        for sigma in (0., -1., np.nan, np.inf):
            self.assertEqual(upper_limit_log_likelihood(
                [1.], constant_model, (self.x, np.array([3.]), np.array([sigma]))), -np.inf)

    def test_optional_empty_and_additive_likelihood(self):
        args = ([2.], constant_model, self.x, np.array([2.]), np.array([0.2]))
        old = log_likelihood(*args)
        empty = ((np.array([]), np.array([])), np.array([]), np.array([]))
        self.assertEqual(log_likelihood(*args, empty), old)
        self.assertEqual(log_likelihood(*args, self.upper), old + log_ndtr(1.))
        # The sampler's log10 normalization parameter must be converted first.
        actual = log_probability([np.log10(2.)], constant_model, ["f0"],
                                 self.x, np.array([2.]), np.array([0.2]), [(-2., 2.)], self.upper)
        self.assertAlmostEqual(actual, old + log_ndtr(1.))

    def test_selection_units_disabled_and_empty(self):
        cfg = {"burst": {"z": 1., "fitstart": 0.5},
               "fit": {"min_freq": 10., "max_rest_freq": 200.}}
        _, _, _, selected, excluded, _ = prepare_fit_data(self.frame, cfg)
        self.assertEqual(selected.index.tolist(), [7])
        self.assertEqual(excluded.index.tolist(), [10, 12])
        x, limits, sigma = prepare_upper_limit_data(selected, cfg)
        np.testing.assert_array_equal(x, [[2.], [10.]])
        np.testing.assert_allclose(limits, [6e-6])
        np.testing.assert_allclose(sigma, [2e-6])
        self.assertIsNone(prepare_upper_limit_data(selected.iloc[:0], cfg))
        cfg["fit"]["fit_upper_limits"] = False
        self.assertIsNone(prepare_upper_limit_data(selected, cfg))
        cfg["fit"]["fit_upper_limits"] = True
        cfg["burst"]["fitstart"] = 2.5
        with self.assertRaisesRegex(ValueError, "No detections remain"):
            prepare_fit_data(self.frame, cfg)

    def test_invalid_selected_rows_report_indices(self):
        for column in ("obsdate", "freq", "rms"):
            for value in (0., np.nan, np.inf, "invalid"):
                frame = self.frame.loc[[7]].astype(object)
                frame.loc[7, column] = value
                with self.subTest(column=column, value=value):
                    with self.assertRaisesRegex(ValueError, r"indices \[7\]"):
                        prepare_upper_limit_data(frame, {"fit": {}})
                    self.assertIsNone(prepare_upper_limit_data(frame, {"fit": {"fit_upper_limits": False}}))

    def test_default_and_boolean_config(self):
        cfg = config()
        del cfg["fit"]["fit_upper_limits"]
        self.assertTrue(normalize_config(cfg)["fit"]["fit_upper_limits"])
        for value in ("false", None, 1):
            cfg["fit"]["fit_upper_limits"] = value
            with self.assertRaisesRegex(ValueError, "fit.fit_upper_limits"):
                normalize_config(cfg)

    def test_extinction_band_eligibility_and_corrected_model(self):
        cfg = config()
        cfg["burst"].update(z=1., fitstart=0.5)
        cfg["fit"].update(fit_above_max_rest_freq=True, max_rest_freq=200.)
        frame = pd.DataFrame({
            "obsdate": [1., 2., 3., 4., 1., 2., 3.],
            "freq": [100.] * 4 + [200.] * 3,
            "flux": [100.] * 3 + [-1.] + [100.] * 2 + [-1.],
            "err": [10.] * 3 + [-1.] + [10.] * 2 + [-1.],
            "rms": [2.] * 7, "instrument": ["other"] * 7,
        })
        prepared = prepare_fit_data(frame, cfg)
        self.assertEqual(prepared[3].index.tolist(), [3])
        self.assertEqual(prepared[4].index.tolist(), [4, 5, 6])
        upper = prepare_upper_limit_data(prepared[3], cfg)
        keys, p0, _ = build_param_vector(cfg)
        theta = theta_to_physical(p0, keys)
        theta[keys.index("extinction_correction_1")] = 4.
        intrinsic_cfg = copy.deepcopy(cfg)
        intrinsic_cfg["fit"]["extinction_bands"] = {}
        observed = make_model(intrinsic_cfg)(theta, upper[0]) / 4.
        expected = np.sum(log_ndtr((upper[1] - observed) / upper[2]))
        self.assertAlmostEqual(upper_limit_log_likelihood(theta, make_model(cfg), upper), expected)

    def test_report_keeps_detection_statistics_separate(self):
        cfg = {"fit": {"param_keys": ["f0"]}}
        args = (cfg, np.array([[2.]]), self.x, np.array([2.]), np.array([0.2]))
        with patch("grbfit.run.make_model", return_value=constant_model):
            old = calculate_goodness_of_fit(*args)
            metrics = calculate_goodness_of_fit(*args, self.upper)
            cfg["fit"]["fit_upper_limits"] = False
            disabled = calculate_goodness_of_fit(*args, self.upper)
        for key in ("chisq", "AIC", "BIC", "DOF", "ndata"):
            self.assertEqual(metrics[key], old[key])
        self.assertEqual(metrics["nupper"], 1)
        self.assertAlmostEqual(metrics["upper_limit_log_likelihood"], log_ndtr(1.))
        self.assertEqual(disabled["nupper"], 0)
        self.assertEqual(disabled["upper_limit_log_likelihood"], 0.)
        report = format_goodness_of_fit_report(*args, metrics)
        self.assertIn("Detection-only", report)
        self.assertIn("Upper limits used: 1", report)

    def test_sampler_passes_limits_and_honors_disable(self):
        for enabled in (True, False):
            cfg = {"fit": {"fit_upper_limits": enabled, "mcmc_mode": "fixed",
                           "burn_in": 1, "nsteps": 1}}
            with patch("grbfit.fit.build_param_vector", return_value=(["f0"], [0.], [(-2., 2.)])), \
                    patch("grbfit.fit.make_model", return_value=constant_model), \
                    patch("grbfit.fit.emcee.EnsembleSampler") as sampler:
                sampler.return_value.acceptance_fraction = np.array([0.5])
                sampler.return_value.iteration = 1
                sampler.return_value.get_autocorr_time.return_value = np.array([1.])
                run_mcmc(cfg, self.x, np.array([2.]), np.array([0.2]), upper_limits=self.upper)
                passed = sampler.call_args.kwargs["args"][-1]
                self.assertIs(passed, self.upper if enabled else None)

    def test_restrictive_limit_suppresses_high_flux_posterior(self):
        # Integrate a one-parameter posterior on a fixed uniform-prior grid.
        grid = np.linspace(0.01, 10., 2000)
        y, err = np.array([5.]), np.array([3.])
        old = np.array([log_likelihood([f], constant_model, self.x, y, err) for f in grid])
        new = np.array([log_likelihood([f], constant_model, self.x, y, err, self.upper) for f in grid])
        def high_flux_mass(logp):
            weights = np.exp(logp - logp.max())
            return weights[grid > 5.].sum() / weights.sum()
        self.assertLess(high_flux_mass(new), high_flux_mass(old) / 10.)


if __name__ == "__main__":
    unittest.main()
