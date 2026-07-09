import subprocess
import sys
from pathlib import Path


def test_regime_gallery_example_generates_expected_figures(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "model_regime_gallery"
        / "plot_regime_gallery.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--formats",
            "png",
            "--n-points",
            "80",
            "--no-show",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    expected = [
        "forward_shock_spectra.png",
        "forward_shock_light_curves.png",
        "forward_shock_absorption_factor.png",
        "reverse_shock_spectra.png",
        "reverse_shock_light_curves.png",
    ]
    for filename in expected:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
