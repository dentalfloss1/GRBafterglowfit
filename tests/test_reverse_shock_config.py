import csv
import os
import tempfile
import unittest
import warnings

import numpy as np

from grbfit.artifacts import build_model_fit_artifact
from grbfit.fit import build_param_vector, make_model, uses_log10_sampling
from grbfit.run import normalize_config, write_standardized_fit_csv


def base_config():
    initial = {
        "f0": 1e-3,
        "f0_rev": 5e-4,
        "nua0_rev": 10.0,
        "num0_rev": 100.0,
        "nuc0_rev": 1e8,
        "nua_0": 10.0,
        "num_0": 100.0,
        "nuc_0": 1e8,
        "t_j": None,
    }
    return {
        "burst": {"t0": 1.0, "t0_rev": 0.05, "fitstart": 0.01},
        "data": {},
        "model": {"type": "forward_reverse", "k": 2, "p": 2.2},
        "fit": {
            "initial_guess": initial,
            "bounds": {
                key: [value, value]
                for key, value in initial.items()
                if value is not None
            },
            "max_rest_freq": 2.47e6,
            "fit_xrt": False,
        },
    }


class ReverseShockConfigTests(unittest.TestCase):
    def test_old_config_defaults_to_fixed_thick_shell(self):
        cfg = normalize_config(base_config())

        self.assertEqual(cfg["model"]["reverse_shell"], "thick")
        self.assertFalse(cfg["model"]["fit_g"])
        self.assertNotIn("g", cfg["fit"]["initial_guess"])

    def test_fixed_thin_shell_uses_midpoint_g(self):
        cfg = base_config()
        cfg["model"]["reverse_shell"] = "thin"
        cfg = normalize_config(cfg)
        keys, _, _ = build_param_vector(cfg)

        self.assertNotIn("g", keys)
        self.assertEqual(cfg["fit"]["fixed_params"]["g"], 1.0)
        self.assertEqual(cfg["fit"]["bounds"]["g"], [1.0, 1.0])

    def test_fitted_thin_shell_derives_linear_g_bounds(self):
        for k, expected in ((0, (1.5, 2.5, 3.5)), (2, (0.5, 1.0, 1.5))):
            cfg = base_config()
            cfg["model"].update(
                {"k": k, "reverse_shell": "thin", "fit_g": True}
            )
            cfg = normalize_config(cfg)
            keys, p0, bounds = build_param_vector(cfg)
            g_index = keys.index("g")
            lower, midpoint, upper = expected

            self.assertFalse(uses_log10_sampling("g"))
            self.assertEqual(p0[g_index], midpoint)
            np.testing.assert_allclose(bounds[g_index], [lower, upper])

    def test_fit_g_warns_and_is_ignored_for_thick_shell(self):
        cfg = base_config()
        cfg["model"]["fit_g"] = True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = normalize_config(cfg)

        self.assertTrue(any("fit_g is ignored" in str(item.message) for item in caught))
        self.assertFalse(cfg["model"]["fit_g"])
        self.assertNotIn("g", cfg["fit"]["initial_guess"])

    def test_invalid_shell_and_k_are_rejected(self):
        invalid_shell = base_config()
        invalid_shell["model"]["reverse_shell"] = "medium"
        with self.assertRaisesRegex(ValueError, "reverse_shell"):
            normalize_config(invalid_shell)

        invalid_k = base_config()
        invalid_k["model"]["k"] = 1.7
        with self.assertRaisesRegex(ValueError, "model.k"):
            normalize_config(invalid_k)

    def test_fitted_g_is_saved_in_artifact(self):
        cfg = base_config()
        cfg["model"].update(
            {"reverse_shell": "thin", "fit_g": True}
        )
        cfg = normalize_config(cfg)
        keys, _, _ = build_param_vector(cfg)
        samples = np.array([[1.0]])
        self.assertEqual(keys, ["g"])

        artifact = build_model_fit_artifact(cfg, samples, {})

        self.assertEqual(artifact["model"]["reverse_shell"], "thin")
        self.assertTrue(artifact["model"]["fit_g"])
        self.assertEqual(artifact["fit"]["parameters"]["g"]["status"], "fitted")
        self.assertEqual(artifact["fit"]["parameters"]["g"]["value"], 1.0)

    def test_fitted_g_changes_combined_model(self):
        cfg = base_config()
        cfg["model"].update(
            {"reverse_shell": "thin", "fit_g": True}
        )
        cfg = normalize_config(cfg)
        keys, _, _ = build_param_vector(cfg)
        model = make_model(cfg)
        ivar = (np.array([1.0]), np.array([10.0]))

        low_g = model(np.array([0.6]), ivar)
        high_g = model(np.array([1.4]), ivar)

        self.assertEqual(keys, ["g"])
        self.assertFalse(np.allclose(low_g, high_g, rtol=1e-6, atol=0))

    def test_standardized_csv_includes_g(self):
        cfg = base_config()
        cfg["model"]["reverse_shell"] = "thin"
        cfg = normalize_config(cfg)
        build_param_vector(cfg)
        goodness = {"redchisq": 1.0, "DOF": 1, "AIC": 2.0, "BIC": 3.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fit_summary.csv")
            write_standardized_fit_csv(cfg, np.empty((1, 0)), goodness, path)
            with open(path, newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(float(row["g"]), 1.0)
        self.assertEqual(float(row["gerrneg"]), -1.0)
        self.assertEqual(float(row["gerrpos"]), -1.0)


if __name__ == "__main__":
    unittest.main()
