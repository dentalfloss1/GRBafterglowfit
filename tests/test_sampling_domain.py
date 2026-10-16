import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from grbfit.fit import build_param_vector, log_probability, make_model, theta_to_physical
from grbfit.models import UnsupportedBreakOrderingError
from grbfit.run import create_template_config, normalize_config


def template():
    return yaml.safe_load((Path(__file__).parents[1] / "src/grbfit/config_template.yaml").read_text())


class SamplingDomainTests(unittest.TestCase):
    def test_generated_templates_initialize_both_models_with_ordered_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            for fallback in (False, True):
                path = Path(directory) / f"config-{fallback}.yaml"
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if fallback:
                        stack.enter_context(patch("grbfit.run.resources.files", side_effect=FileNotFoundError))
                    create_template_config(path)
                generated = yaml.safe_load(path.read_text())
                self.assertEqual(generated["burst"], template()["burst"])
                for section in ("initial_guess", "bounds"):
                    self.assertEqual(generated["fit"][section], template()["fit"][section])
                self.assertEqual(generated["burst"]["t0"], .05)
                self.assertEqual(generated["burst"]["t0_rev"], .05)
                for a, c in (("nua_0", "nuc_0"), ("nua0_rev", "nuc0_rev")):
                    self.assertEqual(float(generated["fit"]["bounds"][a][0]), 1.)
                    self.assertLessEqual(float(generated["fit"]["bounds"][a][1]),
                                         float(generated["fit"]["bounds"][c][0]))
                for kind in ("forward_only", "forward_reverse"):
                    cfg = copy.deepcopy(generated)
                    cfg["model"]["type"] = kind
                    cfg["fit"]["initial_guess"]["t_j"] = .3
                    cfg = normalize_config(cfg)
                    keys, p0, bounds = build_param_vector(cfg)
                    x = (np.array([.01, .05, .1, 1.]), np.array([10., 100., 1e4, 1e8]))
                    model = make_model(cfg)
                    flux = model(theta_to_physical(p0, keys), x)
                    self.assertTrue(np.all(np.isfinite(flux) & (flux > 0)))
                    self.assertTrue(np.isfinite(log_probability(p0, model, keys, x, flux, .1*flux, bounds)))

    def test_bad_initial_ordering_is_rejected_and_valid_proposal_still_evaluates(self):
        for kind, absorption, cooling in (("forward_only", "nua_0", "nuc_0"),
                                          ("forward_reverse", "nua0_rev", "nuc0_rev")):
            cfg = template()
            cfg["model"]["type"] = kind
            cfg = normalize_config(cfg)
            keys, p0, _ = build_param_vector(cfg)
            model = make_model(cfg)
            x = (np.array([.05]), np.array([100.]))
            flux = model(theta_to_physical(p0, keys), x)
            bounds = np.tile([-20., 20.], (len(keys), 1))
            bad = p0.copy()
            bad[keys.index(absorption)] = 9
            bad[keys.index(cooling)] = 8
            with self.assertRaises(UnsupportedBreakOrderingError):
                model(theta_to_physical(bad, keys), x)
            self.assertEqual(log_probability(bad, model, keys, x, flux, .1*flux, bounds), -np.inf)
            self.assertTrue(np.isfinite(log_probability(p0, model, keys, x, flux, .1*flux, bounds)))

    def test_later_reverse_crossing_rejected_for_detections_and_upper_limits(self):
        cfg = template()
        cfg["model"]["type"] = "forward_reverse"
        cfg["fit"]["initial_guess"].update(nua0_rev=10., nuc0_rev=20., num0_rev=1000.)
        cfg["fit"]["bounds"]["nuc0_rev"] = [1., 1e12]
        cfg = normalize_config(cfg)
        keys, p0, bounds = build_param_vector(cfg)
        model = make_model(cfg)
        early, late = (np.array([.05]), np.array([100.])), (np.array([1.]), np.array([100.]))
        flux = model(theta_to_physical(p0, keys), early)
        with self.assertRaisesRegex(UnsupportedBreakOrderingError, "after nu_a/nu_c"):
            model(theta_to_physical(p0, keys), late)
        self.assertEqual(log_probability(p0, model, keys, late, flux, .1*flux, bounds), -np.inf)
        self.assertEqual(log_probability(p0, model, keys, early, flux, .1*flux, bounds,
                                         (late, flux, .1*flux)), -np.inf)

    def test_unrelated_model_errors_remain_visible(self):
        def broken_model(theta, x):
            raise ValueError("configuration error")

        with self.assertRaisesRegex(ValueError, "configuration error"):
            log_probability([0.], broken_model, ["f0"], None, None, None, [(-1., 1.)])


if __name__ == "__main__":
    unittest.main()
