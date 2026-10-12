"""
Tests for DGM command line interface (CLI).
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import pytest

from dgm.cli import main


def test_cli_generate_command(tmp_path):
    """Verify CLI generate command produces valid, readable GraphML file."""
    out_file = tmp_path / "test_delta.graphml"
    main(["generate", "-N", "20", "-R", "0.3", "-s", "42", "-o", str(out_file)])
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Verify GraphML content is a valid NetworkX graph
    G = nx.read_graphml(str(out_file))
    assert G.number_of_nodes() >= 10
    assert G.number_of_edges() > 0


def test_cli_ensemble_command(tmp_path):
    """Verify CLI ensemble command produces valid, readable CSV file with metric summaries."""
    out_csv = tmp_path / "test_ensemble.csv"
    main(["ensemble", "-N", "20", "-R", "0.5", "-n", "3", "-s", "42", "-o", str(out_csv)])
    assert out_csv.exists()
    assert out_csv.stat().st_size > 0

    # Verify CSV file is parseable and contains summary rows
    df = pd.read_csv(out_csv)
    assert len(df) > 0


def test_cli_no_args_prints_help(capsys):
    """Verify invoking CLI with no arguments prints help text cleanly."""
    main([])
    captured = capsys.readouterr()
    assert "DeltaGraphModel" in captured.out or "usage:" in captured.out


def test_cli_invalid_argument_exits():
    """Verify passing unrecognized flag causes CLI to exit cleanly with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--unrecognized-argument-xyz"])
    assert exc_info.value.code == 2
