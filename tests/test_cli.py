"""
Tests for DGM command line interface (CLI).
"""

import tempfile
from pathlib import Path
from dgm.cli import main


def test_cli_generate_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_delta.graphml"
        # Run CLI generate command
        main(["generate", "-N", "20", "-R", "0.3", "-s", "42", "-o", str(out_file)])
        assert out_file.exists()
        assert out_file.stat().st_size > 0


def test_cli_ensemble_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "test_ensemble.csv"
        # Run CLI ensemble command
        main(["ensemble", "-N", "20", "-R", "0.5", "-n", "3", "-s", "42", "-o", str(out_csv)])
        assert out_csv.exists()
        assert out_csv.stat().st_size > 0
