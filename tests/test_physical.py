import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grbfit.artifacts import write_posterior_samples
from grbfit.models import _forward_shock_spectrum, FS_SELF_ABSORBED_SLOW
from grbfit.physical import (
    MPC, derive_physical_posterior, jet_half_angle, luminosity_distance_cm,
    main, predict_observables,
)


def synthetic(k=0, regime="slow", p=2.2, jet=False):
    times = {(0, "slow"): 1, (0, "fast"): .001, (0, "self_absorbed_slow"): 100,
             (2, "slow"): 2, (2, "fast"): 1, (2, "self_absorbed_slow"): 100}
    t0 = times[k, regime]
    physical = np.array([[1e52, 1., .1, .1], [1.01e52, 1.02, .101, .099]])
    samples = []
    for energy, density, ee, eb in physical:
        row = predict_observables(energy, density, ee, eb, p=p, k=k, time_days=t0,
                                  z=1, distance_cm=1e28, regime=regime)
        if regime == "self_absorbed_slow":
            row[0] /= (row[2]/row[1])**((2-p)/2)
        samples.append([*row, p] + ([t0*2] if jet else []))
    keys = ["f0", "nua_0", "num_0", "nuc_0", "p"] + (["t_j"] if jet else [])
    cfg = {"burst": {"z": 1, "t0": t0}, "model": {"k": k, "type": "forward_only"},
           "fit": {"param_keys": keys, "fixed_params": {}}}
    metadata = {"config": cfg, "fixed_params": {}, "sampler": {"converged": True}}
    return np.array(samples), keys, metadata, physical


class PhysicalTests(unittest.TestCase):
    def test_published_table_normalizations(self):
        # Independent rounded benchmarks from Tables 2.6 / 2.7, at p=2.2,
        # z=1, t=1 d, D=1e28 cm, E=1e52 erg, n or A*=1, ee=eB=0.1.
        for k, expected in ((0, (.0213, 77.5, 898., 5.98e4, 1.25, 113.)),
                            (2, (.0608, 516., 1850., 997., 92.3, 438.))):
            for regime, absorption in (("slow", expected[1]), ("fast", expected[4]),
                                       ("self_absorbed_slow", expected[5])):
                with self.subTest(k=k, regime=regime):
                    actual = predict_observables(1e52, 1, .1, .1, p=2.2, k=k, time_days=1,
                                                 z=1, distance_cm=1e28, regime=regime)
                    np.testing.assert_allclose(actual, [expected[0], absorption, expected[2], expected[3]], rtol=.015)

    def test_all_regimes_recover_parameters_and_joint_rows(self):
        for k in (0, 2):
            for regime in ("slow", "fast", "self_absorbed_slow"):
                for jet in (False, True):
                    with self.subTest(k=k, regime=regime, jet=jet):
                        samples, keys, metadata, expected = synthetic(k, regime, jet=jet)
                        out, names, status, report = derive_physical_posterior(samples, keys, metadata, distance_cm=1e28)
                        np.testing.assert_allclose(out[:, :4], expected, rtol=2e-12)
                        self.assertEqual(status.tolist(), ["valid", "valid"])
                        self.assertEqual(report["valid_draws"], 2)
                        self.assertEqual(names[1], "n" if k == 0 else "A_star")
                        self.assertEqual(len(names), 7 if jet else 4)
                        if jet:
                            np.testing.assert_allclose(out[:, 5], np.degrees(out[:, 4]))
                            np.testing.assert_allclose(out[:, 6], out[:, 0]*(1-np.cos(out[:, 4])))

    def test_independent_rounded_observable_inversion(self):
        samples, keys, metadata, _ = synthetic()
        samples[:] = [.0213, 77.5, 898., 5.98e4, 2.2]
        out, _, status, _ = derive_physical_posterior(samples, keys, metadata, distance_cm=1e28)
        self.assertEqual(status[0], "valid")
        np.testing.assert_allclose(out[0], [1e52, 1, .1, .1], rtol=.03)

    def test_self_absorbed_normalization_matches_fitted_thin_asymptote(self):
        p, f0, nua, num, nuc, nu = 2.7, .01, 100., 1., 1e12, 1e6
        flux = _forward_shock_spectrum(FS_SELF_ABSORBED_SLOW, nu, f0, nua, num, nuc, p)
        fmax = f0*(num/nua)**((2-p)/2)
        self.assertAlmostEqual(flux/(fmax*(nu/num)**(-(p-1)/2)), 1., places=8)

    def test_variable_p_and_fixed_p_jet_reverse_nuisance(self):
        samples, keys, metadata, expected = synthetic(p=2.5, jet=True)
        # p and tj fixed; an arbitrary reverse-shock column must not enter inversion.
        metadata["fixed_params"] = {"p": 2.5, "t_j": 2.}
        samples = np.column_stack([samples[:, :4], [.1, .2]])
        keys = keys[:4] + ["f0_rev"]
        metadata["config"]["model"]["type"] = "forward_reverse"
        out, _, status, _ = derive_physical_posterior(samples, keys, metadata, distance_cm=1e28)
        np.testing.assert_allclose(out[:, :4], expected, rtol=2e-12)
        self.assertTrue(np.all(status == "valid"))

    def test_domain_statuses_preserve_rows_and_do_not_clip(self):
        samples, keys, metadata, _ = synthetic(jet=True)
        samples = np.repeat(samples[:1], 6, axis=0)
        samples[1, keys.index("p")] = 2
        samples[2, keys.index("t_j")] = .5
        samples[3, 1:4] = [1e9, 1, 1e6]
        samples[4, 0] = -1
        # Deliberately inject an algebraic solution with ee + eB > 1.
        samples[5, :4] = predict_observables(1e52, 1, .99, .02, p=2.2, k=0, time_days=1,
                                            z=1, distance_cm=1e28, regime="slow")
        out, _, status, report = derive_physical_posterior(samples, keys, metadata, distance_cm=1e28)
        self.assertEqual(status.tolist(), ["valid", "unsupported_p", "reference_not_prejet",
                                         "unsupported_ordering", "invalid_observables", "unphysical_energy_fractions"])
        self.assertTrue(np.all(np.isnan(out[1:5])))
        np.testing.assert_allclose(out[5, 2:4], [.99, .02])
        self.assertEqual(report["valid_draws"], 1)
        self.assertAlmostEqual(report["excluded_fraction"], 5/6)
        self.assertAlmostEqual(report["summary"]["epsilon_e"]["median"], .1)

    def test_required_inputs_and_fixed_observables(self):
        samples, keys, metadata, _ = synthetic()
        for field, value, message in (("z", None, "Redshift"), ("z", 0, "redshift"), ("t0", -1, "t0")):
            modified = copy.deepcopy(metadata)
            modified["config"]["burst"][field] = value
            with self.assertRaisesRegex(ValueError, message):
                derive_physical_posterior(samples, keys, modified, distance_cm=1e28)
        metadata["config"]["model"]["k"] = 1
        with self.assertRaisesRegex(ValueError, "k=0"):
            derive_physical_posterior(samples, keys, metadata)
        metadata["config"]["model"]["k"] = 0
        metadata["fixed_params"]["nuc_0"] = samples[0, 3]
        keys.remove("nuc_0")
        samples = np.delete(samples, 3, axis=1)
        with self.assertRaisesRegex(ValueError, "Fixed observables"):
            derive_physical_posterior(samples, keys, metadata, distance_cm=1e28)
        _, _, _, report = derive_physical_posterior(samples, keys, metadata, distance_cm=1e28,
                                                    allow_fixed_observables=True)
        self.assertEqual(report["fixed_observables"], ["nuc_0"])
        self.assertTrue(report["warnings"])

    def test_distance_and_jet_scalings(self):
        self.assertAlmostEqual(luminosity_distance_cm(1)/MPC, 6607.657611774935, places=6)
        for z in (.1, 1, 6):
            self.assertAlmostEqual(luminosity_distance_cm(z, omega_m=0)/MPC,
                                   (1+z)*299792.458/70*z, places=7)
        for k in (0, 2):
            theta = jet_half_angle(1e52, 1, 1, 1, k)
            self.assertAlmostEqual(jet_half_angle(1e52, 2, 1, 1, k)/theta, 2**(1/(8-2*k)))
            self.assertAlmostEqual(jet_half_angle(1e52, 1, 2, 1, k)/theta, 2**((3-k)/(8-2*k)))
            self.assertAlmostEqual(jet_half_angle(1e52, 1, 1, 3, k)/theta, 2**(-(3-k)/(8-2*k)))
        # Familiar ISM ~0.16 rad coefficient for t_rest=1 day, E52/n=1.
        self.assertAlmostEqual(jet_half_angle(1e52, 1, 2, 1, 0), .1628, places=3)

    def test_cli_archive_and_summary(self):
        samples, keys, metadata, _ = synthetic(jet=True)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            source = Path(directory)/"input.npz"
            write_posterior_samples(metadata["config"], samples, {"thin": 1, "converged": False}, source)
            original = source.read_bytes()
            main([str(source), "--output-dir", directory, "--distance-cm", "1e28"])
            self.assertEqual(source.read_bytes(), original)
            report = json.loads((Path(directory)/"physical_summary.json").read_text())
            self.assertEqual(report["assumptions"]["Y"], 0)
            self.assertEqual(report["valid_draws"], 2)
            self.assertIn("convergence", report["warnings"][0])
            with np.load(Path(directory)/"physical_samples.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["samples"].shape, (2, 7))
                self.assertEqual(archive["source_sample_index"].tolist(), [0, 1])
                self.assertEqual(archive["status"].tolist(), ["valid", "valid"])
                self.assertEqual(json.loads(archive["metadata_json"].item()), report)
            samples[:, keys.index("p")] = 1.9
            write_posterior_samples(metadata["config"], samples, {"thin": 1}, source)
            with self.assertRaises(SystemExit) as raised:
                main([str(source), "--output-dir", directory, "--distance-cm", "1e28"])
            self.assertEqual(raised.exception.code, 1)
            report = json.loads((Path(directory)/"physical_summary.json").read_text())
            self.assertEqual(report["summary"], {})
            self.assertEqual(report["status_counts"], {"unsupported_p": 2})


if __name__ == "__main__":
    unittest.main()
