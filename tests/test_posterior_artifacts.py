import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from grbfit.artifacts import load_posterior_samples, write_posterior_samples
from grbfit.fit import build_param_vector, samples_to_physical
from grbfit.run import main, normalize_config


def example_config():
    template = Path(__file__).parents[1] / "src/grbfit/config_template.yaml"
    return normalize_config(yaml.safe_load(template.read_text()))


class PosteriorArtifactTests(unittest.TestCase):
    def test_round_trip_joint_draws_and_metadata(self):
        for reverse in (False, True):
            for fitted_p in (False, True):
                for jet in ("absent", "fixed", "fitted"):
                    with self.subTest(reverse=reverse, fitted_p=fitted_p, jet=jet):
                        cfg = example_config()
                        cfg["model"]["type"] = "forward_reverse" if reverse else "forward_only"
                        cfg["burst"]["z"] = 1.2
                        if fitted_p:
                            cfg["fit"]["bounds"]["p"] = [2.0, 3.0]
                        if jet != "absent":
                            cfg["fit"]["initial_guess"]["t_j"] = 5.0
                            cfg["fit"]["bounds"]["t_j"] = [5.0, 5.0] if jet == "fixed" else [1.0, 10.0]
                        # A fixed break and an extinction parameter must also survive.
                        cfg["fit"]["bounds"]["nuc_0"] = [8e8, 8e8]
                        cfg["fit"]["extinction_bands"] = {"extinction_correction_1": 5e6}
                        cfg["fit"]["extinction_band_frequencies"] = {"extinction_correction_1": [5e6, 5.01e6]}
                        keys, p0, _ = build_param_vector(cfg)
                        samples = samples_to_physical(np.stack([p0 + 0.02, p0, p0 - 0.01]), keys)
                        original = copy.deepcopy(cfg)
                        diagnostics = {"thin": 3, "converged": np.bool_(False), "tau_max": np.nan,
                                       "burn_in": 100, "production_steps": 300}
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "posterior_samples.npz"
                            write_posterior_samples(cfg, samples, diagnostics, path)
                            with np.load(path, allow_pickle=False) as archive:
                                self.assertEqual(archive["param_keys"].tolist(), keys)
                                self.assertEqual(archive["metadata_json"].shape, ())
                            loaded, names, metadata = load_posterior_samples(path)
                        np.testing.assert_array_equal(loaded, samples)
                        self.assertEqual(names, keys)
                        self.assertEqual(metadata["config"], cfg)
                        self.assertEqual(cfg, original)
                        self.assertEqual(metadata["fixed_params"], cfg["fit"]["fixed_params"])
                        self.assertEqual(metadata["parameter_units"]["f0"], "Jy")
                        self.assertEqual(metadata["parameter_units"]["nuc_0"], "GHz")
                        self.assertEqual(metadata["parameter_units"]["p"], "dimensionless")
                        self.assertEqual(metadata["parameter_units"]["extinction_correction_1"], "dimensionless")
                        if jet != "absent":
                            self.assertEqual(metadata["parameter_units"]["t_j"], "d")
                        self.assertEqual(metadata["sampler"]["thin"], 3)
                        self.assertIs(metadata["sampler"]["converged"], False)
                        self.assertIsNone(metadata["sampler"]["tau_max"])
                        self.assertTrue(metadata["burn_in_removed"])

    def test_rejects_malformed_archives(self):
        cfg = example_config()
        keys, p0, _ = build_param_vector(cfg)
        samples = samples_to_physical(p0[None, :], keys)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posterior_samples.npz"
            write_posterior_samples(cfg, samples, {"thin": 1}, path)
            with np.load(path, allow_pickle=False) as archive:
                valid = dict(archive)
            metadata = json.loads(valid["metadata_json"].item())
            cases = [
                ({"samples": samples[0]}, "matrix"),
                ({"samples": samples[:, :-1]}, "column"),
                ({"samples": samples[:0]}, "nonempty"),
                ({"samples": samples * np.nan}, "finite"),
                ({"samples": samples.astype(object)}, "Object arrays"),
                ({"param_keys": np.asarray(keys[::-1])}, "column names"),
                ({"param_keys": np.asarray([keys[0]] * len(keys))}, "unique"),
                ({"metadata_json": np.asarray("not json")}, "Invalid posterior archive"),
                ({"metadata_json": np.asarray("[]")}, "format"),
                ({"metadata_json": np.asarray(json.dumps({**metadata, "format_version": 2}))}, "format_version"),
                ({"metadata_json": np.asarray(json.dumps({**metadata, "fixed_params": {"p": None}}))}, "fixed parameters"),
                ({"metadata_json": np.asarray(json.dumps({**metadata, "parameter_units": {}}))}, "units"),
                ({"metadata_json": np.asarray(json.dumps({**metadata, "sampler": {"thin": 0}}))}, "thinning"),
            ]
            for changes, message in cases:
                with self.subTest(message=message):
                    np.savez_compressed(path, **{**valid, **changes})
                    with self.assertRaisesRegex(ValueError, message):
                        load_posterior_samples(path)
            np.savez_compressed(path, samples=samples)
            with self.assertRaisesRegex(ValueError, "Invalid posterior archive"):
                load_posterior_samples(path)

    def test_pipeline_saves_retained_linear_draws_before_plotting(self):
        for quick in (False, True):
            with self.subTest(quick=quick), tempfile.TemporaryDirectory() as directory:
                cfg = example_config()
                keys, p0, _ = build_param_vector(cfg)
                chain = np.stack([p0 + offset for offset in (0.0, 0.01, 0.02, 0.03)])[:, None, :]
                sampler = Mock()
                sampler.grbfit_diagnostics = {"thin": 2, "converged": False, "production_steps": 4}
                sampler.get_chain.side_effect = lambda discard, thin, flat: chain[thin - 1::thin].reshape(-1, len(keys))

                def run_sampler(resolved, *args, **kwargs):
                    build_param_vector(resolved)
                    sampler.grbfit_diagnostics["mode"] = resolved["fit"]["mcmc_mode"]
                    sampler.grbfit_diagnostics["burn_in"] = resolved["fit"]["burn_in"]
                    return keys, sampler

                def plot_failure(*args):
                    self.assertTrue(Path("posterior_samples.npz").exists())
                    raise RuntimeError("plot failed")

                old_directory = os.getcwd()
                try:
                    os.chdir(directory)
                    Path("config.yaml").write_text(yaml.safe_dump(cfg))
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        for name in ("load_data", "summarize_chain", "calculate_goodness_of_fit", "print_goodness_of_fit_report"):
                            stack.enter_context(patch(f"grbfit.run.{name}"))
                        stack.enter_context(patch("grbfit.run.prepare_fit_data", return_value=(None,) * 6))
                        stack.enter_context(patch("grbfit.run.prepare_upper_limit_data", return_value=None))
                        stack.enter_context(patch("grbfit.run.run_mcmc", side_effect=run_sampler))
                        stack.enter_context(patch("grbfit.run.plot_corner", side_effect=plot_failure))
                        with self.assertRaisesRegex(RuntimeError, "plot failed"):
                            main(["--quick"] if quick else [])
                    loaded, names, metadata = load_posterior_samples()
                    np.testing.assert_array_equal(loaded, samples_to_physical(chain[1::2].reshape(-1, len(keys)), keys))
                    self.assertEqual(names, keys)
                    sampler.get_chain.assert_called_once_with(discard=0, thin=2, flat=True)
                    self.assertEqual(metadata["sampler"]["thin"], 2)
                    if quick:
                        self.assertEqual(metadata["config"]["fit"]["nsteps"], 300)
                        self.assertEqual(metadata["sampler"]["burn_in"], 100)
                        self.assertEqual(metadata["sampler"]["mode"], "fixed")
                finally:
                    os.chdir(old_directory)


if __name__ == "__main__":
    unittest.main()
