import copy
import json
import unittest
from importlib import resources
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from grbfit.artifacts import build_model_fit_artifact
from grbfit.data import prepare_fit_data, X_RAY_MIN_FREQ_GHZ
from grbfit.detectability import _config_and_theta_from_artifact
from grbfit.extinction import group_band_frequencies, correction_factors
from grbfit.fit import (
    build_param_vector, make_model, evaluate_model_components,
    theta_to_physical, samples_to_physical, log_probability,
)
from grbfit.run import normalize_config, calculate_goodness_of_fit, plot_best_fit_residuals, plot_posterior_models


def config():
    cfg = yaml.safe_load(resources.files("grbfit").joinpath("config_template.yaml").read_text())
    cfg["burst"]["z"] = 1
    cfg["burst"]["fitstart"] = 0.5
    cfg["fit"].update(fit_above_max_rest_freq=True, max_rest_freq=200)
    return cfg


def data():
    # Band 100 is on the rest-frame cutoff; band 200 has only two detections.
    return pd.DataFrame({
        "freq": [10, 100, 100, 100, 200, 200, 200, 300, 300, 300, 300, X_RAY_MIN_FREQ_GHZ],
        "obsdate": [1, 1, 2, 3, 1, 2, 3, 0.5, 1, 2, 3, 1],
        "flux": [100] * 6 + [-1] + [100] * 5,
        "err": [10] * 6 + [-1] + [10] * 5,
        "rms": [2] * 12,
        "instrument": ["radio", "other", "radio", "other"] + ["other"] * 7 + ["XRT"],
    })


class ExtinctionTests(unittest.TestCase):
    def test_selection_threshold_cuts_and_xrays(self):
        for fit_xrt in (False, True):
            cfg = config()
            cfg["fit"]["fit_xrt"] = fit_xrt
            original = data()
            prepared = prepare_fit_data(original, cfg)
            (_, nu), y, err, upper, excluded, _ = prepared
            self.assertEqual(cfg["fit"]["extinction_bands"], {
                "extinction_correction_1": 100., "extinction_correction_2": 300.,
            })
            self.assertEqual(len(nu), 7 + fit_xrt)
            self.assertFalse(np.any(nu == 200))
            self.assertEqual(np.count_nonzero(nu == X_RAY_MIN_FREQ_GHZ), int(fit_xrt))
            self.assertTrue(upper.empty)
            self.assertIn(7, excluded.index)  # time equals fitstart
            np.testing.assert_allclose(y, 100e-6)
            np.testing.assert_allclose(err, np.sqrt(104) * 1e-6)
            pd.testing.assert_frame_equal(original, data())
        cfg["fit"]["min_freq"] = 250
        prepare_fit_data(data(), cfg)
        self.assertEqual(list(cfg["fit"]["extinction_bands"].values()), [300.])
        # Removing one post-cut detection makes band 300 ineligible.
        frame = data()
        frame.loc[10, "obsdate"] = 0.1
        prepare_fit_data(frame, cfg)
        self.assertEqual(cfg["fit"]["extinction_bands"], {})

    def test_disabled_and_repeated_preparation(self):
        cfg = config()
        prepare_fit_data(data(), cfg)
        cfg["fit"]["fit_above_max_rest_freq"] = False
        result = prepare_fit_data(data(), cfg)
        np.testing.assert_array_equal(result[0][1], [10])
        self.assertEqual(cfg["fit"]["extinction_bands"], {})
        del cfg["fit"]["fit_above_max_rest_freq"]
        np.testing.assert_array_equal(prepare_fit_data(data(), cfg)[0][1], [10])

    def test_one_percent_grouping_boundary_and_no_chaining(self):
        frequencies = [101.5, 100.5, 100., 101., 103.]
        expected = [[100., 100.5, 101.], [101.5], [103.]]
        self.assertEqual(group_band_frequencies(frequencies), expected)
        self.assertEqual(group_band_frequencies(frequencies[::-1]), expected)
        self.assertEqual(group_band_frequencies([100, 101.00001]), [[100.], [101.00001]])

    def test_rounded_frequencies_share_eligibility_and_correction(self):
        cfg = config()
        cfg["fit"]["max_rest_freq"] = 900000
        frame = data().iloc[:6].copy()
        frame["freq"] = [456000., 460000., 460000., 470000., 470000., 470000.]
        frame.loc[5, ["flux", "err"]] = -1  # Only two detections in the other band.
        prepared = prepare_fit_data(frame, cfg)
        np.testing.assert_array_equal(prepared[0][1], [456000., 460000., 460000.])
        key = "extinction_correction_1"
        self.assertEqual(cfg["fit"]["extinction_bands"], {key: 456000.})
        self.assertEqual(cfg["fit"]["extinction_band_frequencies"], {key: [456000., 460000.]})
        self.assertEqual(prepared[4].index.tolist(), [3, 4, 5])
        keys, p0, _ = build_param_vector(cfg)
        theta = theta_to_physical(p0, keys)
        theta[-1] = 4
        x = (np.ones(4), np.array([456000., 460000., 470000., 455000.]))
        intrinsic_cfg = copy.deepcopy(cfg)
        intrinsic_cfg["fit"]["extinction_bands"] = {}
        intrinsic = make_model(intrinsic_cfg)(theta, x)
        expected = intrinsic / [4, 4, 1, 1]
        np.testing.assert_allclose(make_model(cfg)(theta, x), expected)
        artifact = json.loads(json.dumps(build_model_fit_artifact(cfg, theta[None, :], {})))
        self.assertEqual(artifact["fit"]["parameters"][key]["observed_frequencies_GHz"], [456000., 460000.])
        restored, values = _config_and_theta_from_artifact(artifact)
        np.testing.assert_allclose(evaluate_model_components(restored, values, x)["total"], expected)
        # Reordering input rows must preserve groups and parameter names.
        prepare_fit_data(frame.iloc[::-1], cfg)
        self.assertEqual(cfg["fit"]["extinction_band_frequencies"], {key: [456000., 460000.]})
        cfg["fit"]["fit_above_max_rest_freq"] = False
        cfg["fit"]["max_rest_freq"] = 1e9
        prepare_fit_data(frame, cfg)
        self.assertEqual(cfg["fit"]["extinction_band_frequencies"], {})

    def test_legacy_saved_band_matches_only_original_frequency(self):
        cfg = {"fit": {"extinction_bands": {"extinction_correction_1": 456000.}}}
        np.testing.assert_array_equal(correction_factors(
            cfg, {"extinction_correction_1": 4}, [456000., 460000.]
        ), [4, 1])

    def test_invalid_config(self):
        cases = [
            ("fit", "fit_above_max_rest_freq", "true"),
            ("burst", "z", None), ("burst", "z", float("nan")),
            ("fit", "max_rest_freq", 0), ("fit", "max_rest_freq", float("inf")),
            ("fit", "extinction_correction_bounds", [0, 100]),
            ("fit", "extinction_correction_bounds", [1, 1]),
            ("fit", "extinction_correction_bounds", [1]),
            ("fit", "extinction_correction_initial", 1),
            ("fit", "extinction_correction_initial", float("nan")),
        ]
        for section, key, value in cases:
            with self.subTest(key=key, value=value):
                cfg = config()
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    prepare_fit_data(data(), cfg)
                with self.assertRaises(ValueError):
                    normalize_config(cfg)

    def test_predictions_sampling_and_artifact_round_trip(self):
        for model_type in ("forward_only", "forward_reverse"):
            with self.subTest(model=model_type):
                cfg = config()
                cfg["model"]["type"] = model_type
                cfg = normalize_config(cfg)
                prepare_fit_data(data(), cfg)
                keys, p0, bounds = build_param_vector(cfg)
                theta = theta_to_physical(p0, keys)
                np.testing.assert_allclose(samples_to_physical(p0[None, :], keys)[0], theta)
                np.testing.assert_allclose(bounds[-2:], [[0, 3], [0, 3]])
                theta[-2:] = [4, 10]
                x = (np.array([1., 2., 3., 4.]), np.array([10., 100., 300., 200.]))
                uncorrected = copy.deepcopy(cfg)
                uncorrected["fit"]["extinction_bands"] = {}
                intrinsic = make_model(uncorrected)(theta, x)
                predicted = make_model(cfg)(theta, x)
                np.testing.assert_allclose(predicted, intrinsic / [1, 4, 10, 1])
                components = evaluate_model_components(cfg, theta, x)
                base_components = evaluate_model_components(uncorrected, theta, x)
                for name in components:
                    np.testing.assert_allclose(components[name], base_components[name] / [1, 4, 10, 1])
                np.testing.assert_allclose(components["total"], predicted)
                sampling = p0.copy()
                sampling[-2:] = np.log10([4, 10])
                self.assertAlmostEqual(log_probability(sampling, make_model(cfg), keys, x, predicted, predicted * .1, bounds), 0)
                samples = np.tile(theta, (5, 1))
                metrics = calculate_goodness_of_fit(cfg, samples, x, predicted, predicted * .1)
                self.assertEqual(metrics["nfit"], len(keys))
                self.assertAlmostEqual(metrics["chisq"], 0)
                artifact = json.loads(json.dumps(build_model_fit_artifact(cfg, samples, metrics)))
                info = artifact["fit"]["parameters"][keys[-1]]
                self.assertEqual(info["observed_frequency_GHz"], 300)
                self.assertEqual(info["value"], 10)
                restored, restored_theta = _config_and_theta_from_artifact(artifact)
                np.testing.assert_allclose(evaluate_model_components(restored, restored_theta, x)["total"], predicted)
                with patch("grbfit.run.plt.savefig"), patch("grbfit.run.plt.close"):
                    plot_best_fit_residuals(cfg, samples, x, predicted, predicted * .1, np.array(["other"] * 4))
                    residual_fig = plt.gcf()
                    for ax in residual_fig.axes:
                        for artist in ax.collections:
                            np.testing.assert_allclose(artist.get_offsets()[:, 1], 0, atol=1e-12)
                    empty = data().iloc[:0]
                    plot_posterior_models(cfg, samples, x, predicted, predicted * .1,
                                          empty, empty, np.array(["other"] * 4), n_draws=1)
                    posterior_fig = plt.gcf()
                    # Every time-grid curve must agree with a corrected band model.
                    curve_count = 0
                    for ax in posterior_fig.axes:
                        for line in ax.lines:
                            times = np.asarray(line.get_xdata())
                            fluxes = np.asarray(line.get_ydata())
                            if len(times) != 300:
                                continue
                            curve_count += 1
                            self.assertTrue(any(np.allclose(
                                fluxes, make_model(cfg)(theta, (times, np.full_like(times, frequency))) * 1e6
                            ) for frequency in x[1]))
                    self.assertGreater(curve_count, 0)
                plt.close(residual_fig)
                plt.close(posterior_fig)



if __name__ == "__main__":
    unittest.main()
