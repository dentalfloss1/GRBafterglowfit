import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from grbfit.data import load_swift_data


XRT_ROW = "86400 1 -1 1e-12 2e-13 -2e-13\n"


class LoadSwiftDataTests(unittest.TestCase):
    def _load(self, section):
        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "xrt.txt"
            filepath.write_text(f"READ TERR 1 2\n! {section}\n{XRT_ROW}")
            return load_swift_data(filepath, 1.71, 1.0)

    def test_loads_pc_section(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = self._load("PC")

        self.assertEqual(caught, [])
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]["instrument"], "XRT")

    def test_warns_when_loading_pc_incbad_section(self):
        with self.assertWarnsRegex(
            UserWarning,
            "You are including XRT data marked as bad",
        ):
            data = self._load("PC_incbad")

        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
