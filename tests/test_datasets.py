"""
Tests for empirical delta dataset structures and loaders.
"""

from dgm.io.datasets import (
    DIDNTLoader,
    DELTA_FORCING_REGIMES,
    get_empirical_deltas,
)


def test_delta_forcing_regimes_table():
    assert len(DELTA_FORCING_REGIMES) == 28
    assert "Mississippi" in DELTA_FORCING_REGIMES
    assert "Niger" in DELTA_FORCING_REGIMES
    assert DELTA_FORCING_REGIMES["Mississippi"]["forcing_type"] == "river_dominated"
    assert DELTA_FORCING_REGIMES["Niger"]["forcing_type"] == "tide_influenced"


def test_didnt_loader_instantiation():
    loader = DIDNTLoader()
    # Check if dataset exists or fallback gracefully
    if loader.is_available():
        deltas = loader.list_available_deltas()
        assert len(deltas) > 0
