"""
Tests for DGM spatial network visualization utilities.
Verifies 2D cartographic plotting, sinuous polyline channel rendering,
Murray's law width hierarchies, node taxonomies, orientation transforms,
and multi-stage comparisons.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import dgm
from dgm.visualization import (
    _infer_node_type,
    _transform_point,
    _transform_polyline,
    plot_network,
    plot_stages,
    plot_tree_vs_network,
)


def test_transform_point_and_polyline():
    """Verify coordinate transformation across cartesian, downward, and upward orientations."""
    x, y = 1000.0, 200.0

    # Cartesian
    cx, cy = _transform_point(x, y, "cartesian")
    assert math.isclose(cx, 1000.0) and math.isclose(cy, 200.0)

    # Downward: x becomes -y (down), y becomes x (lateral)
    dx, dy = _transform_point(x, y, "downward")
    assert math.isclose(dx, 200.0) and math.isclose(dy, -1000.0)

    # Upward: x becomes +y (up), y becomes x (lateral)
    ux, uy = _transform_point(x, y, "upward")
    assert math.isclose(ux, 200.0) and math.isclose(uy, 1000.0)


    # Polyline array
    pts = [[0.0, 0.0], [500.0, 100.0], [1000.0, 0.0]]
    arr_c = _transform_polyline(pts, "cartesian")
    assert np.allclose(arr_c, pts)

    arr_d = _transform_polyline(pts, "downward")
    assert np.allclose(arr_d, [[0.0, 0.0], [100.0, -500.0], [0.0, -1000.0]])


def test_infer_node_type():
    """Verify node taxonomy classification based on in/out degrees."""
    G = nx.DiGraph()
    G.add_node(0)  # Isolated
    G.add_edge(1, 2)  # 1 is apex
    G.add_edge(1, 3)  # 1 bifurcates
    G.add_edge(2, 4)
    G.add_edge(3, 4)  # 4 is confluence
    G.add_edge(4, 5)  # 5 is outlet

    assert _infer_node_type(G, 1) == "apex"
    assert _infer_node_type(G, 4) == "confluence"
    assert _infer_node_type(G, 5) == "outlet"
    assert _infer_node_type(G, 2) == "internal"


def test_plot_network_basic():
    """Verify plot_network creates a plot, renders edges, and returns an Axes object."""
    G = dgm.generate(N=25, R_rt=0.4, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))

    ret_ax = plot_network(G, ax=ax, title="Test Delta Network", show_legend=True, show_stats=True)
    assert ret_ax is ax
    assert len(ax.lines) > 0  # Domain boundaries and channel lines
    plt.close(fig)


def test_plot_network_curved_vs_straight():
    """Verify curved polyline rendering versus straight segment fallback."""
    G = dgm.generate(N=30, R_rt=0.5, enable_curved_centerlines=True, seed=99)

    # 1. Curved = True: polylines should have > 2 vertices
    fig1, ax1 = plt.subplots()
    plot_network(G, ax=ax1, curved=True)
    curved_line_lengths = [len(line.get_xdata()) for line in ax1.lines]
    # At least some lines must have 21 vertices (the default num_polyline_points)
    assert any(length > 2 for length in curved_line_lengths)
    plt.close(fig1)

    # 2. Curved = False: all channel lines must be straight 2-vertex segments
    fig2, ax2 = plt.subplots()
    plot_network(G, ax=ax2, curved=False, show_domain=False)
    straight_line_lengths = [len(line.get_xdata()) for line in ax2.lines]
    assert all(length == 2 for length in straight_line_lengths)
    plt.close(fig2)


def test_plot_network_on_the_fly_curving():
    """Verify that a graph without precomputed polylines can synthesize them on the fly."""
    G_straight = dgm.generate(N=20, R_rt=0.3, enable_curved_centerlines=False, seed=12)
    # Ensure graph initially has no coordinates
    assert not any("coordinates" in d for _, _, d in G_straight.edges(data=True))

    fig, ax = plt.subplots()
    plot_network(G_straight, ax=ax, curved=True)
    # Original graph must remain unmutated
    assert not any("coordinates" in d for _, _, d in G_straight.edges(data=True))
    # But plotted axes should contain multi-vertex polylines
    curved_lengths = [len(line.get_xdata()) for line in ax.lines]
    assert any(length > 2 for length in curved_lengths)
    plt.close(fig)


@pytest.fixture(scope="module")
def sample_curved_graph():
    return dgm.generate(N=20, R_rt=0.4, enable_curved_centerlines=True, seed=7)


@pytest.mark.parametrize("orient", ["cartesian", "downward", "upward"])
@pytest.mark.parametrize("th", ["light", "dark", "clean"])
def test_plot_network_orientations_and_themes(sample_curved_graph, orient, th):
    """Verify that orientations and themes execute without error."""
    fig, ax = plt.subplots()
    plot_network(sample_curved_graph, ax=ax, orientation=orient, theme=th, show_legend=True)
    assert len(ax.lines) > 0
    plt.close(fig)



def test_plot_tree_vs_network():
    """Verify side-by-side tree backbone vs final network comparison figure."""
    G_final, stages = dgm.generate(N=35, R_rt=0.5, enable_curved_centerlines=True, return_stages=True, seed=42)

    fig, (ax1, ax2) = plot_tree_vs_network(
        stages["tree"],
        G_final,
        curved=True,
        title="DGM Tree vs Final Network"
    )

    assert isinstance(fig, plt.Figure)
    assert ax1.get_title().startswith("Stage 1")
    assert ax2.get_title().startswith("Final Network")
    assert np.allclose(ax1.get_xlim(), ax2.get_xlim(), atol=1e-3)
    assert np.allclose(ax1.get_ylim(), ax2.get_ylim(), atol=1e-3)
    plt.close(fig)


def test_plot_stages():
    """Verify multi-panel progressive build-up evolution figure."""
    _G_final, stages = dgm.generate(N=25, R_rt=0.6, enable_curved_centerlines=True, return_stages=True, seed=88)

    fig, axes = plot_stages(stages, curved=True)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 4
    # All axes should have synchronized limits
    x_lim0 = axes[0].get_xlim()
    for ax in axes[1:]:
        assert np.allclose(ax.get_xlim(), x_lim0, atol=1e-3)
    plt.close(fig)


def test_plot_network_edge_cases():
    """Verify robust handling of empty graphs, single-node graphs, and missing attributes."""
    # 1. Empty graph
    G_empty = nx.DiGraph()
    fig1, ax1 = plt.subplots()
    plot_network(G_empty, ax=ax1)
    plt.close(fig1)

    # 2. Single node graph
    G_single = nx.DiGraph()
    G_single.add_node(0, x=0.0, y=0.0)
    fig2, ax2 = plt.subplots()
    plot_network(G_single, ax=ax2)
    plt.close(fig2)

    # 3. Missing width and sinuosity
    G_bare = nx.DiGraph()
    G_bare.add_node(0, x=0.0, y=0.0)
    G_bare.add_node(1, x=100.0, y=50.0)
    G_bare.add_edge(0, 1)
    fig3, ax3 = plt.subplots()
    plot_network(G_bare, ax=ax3, curved=True)
    plt.close(fig3)


def test_cli_image_export(tmp_path):
    """Verify that CLI export with a .png extension saves a valid image file."""
    from dgm.cli import main
    out_img = tmp_path / "cli_delta.png"

    main([
        "generate",
        "-N", "20",
        "-R", "0.4",
        "--curved",
        "-o", str(out_img)
    ])

    assert out_img.exists()
    assert out_img.stat().st_size > 1000


def test_legend_all_types_contains_confluence_orange_diamond():
    """Verify that legends include confluence markers (orange diamond) by default across all plotting functions."""
    # 1. Pure tree backbone has 0 confluences in graph, but legend should include confluence
    G_tree = dgm.generate(N=20, R_rt=0.0, enable_pruning=False, seed=42)
    fig1, ax1 = plt.subplots()
    plot_network(G_tree, ax=ax1, show_legend=True, legend_all_types=True)
    leg1 = ax1.get_legend()
    assert leg1 is not None
    labels1 = [t.get_text() for t in leg1.get_texts()]
    assert any("Confluence" in label for label in labels1)

    # Check marker style and color for confluence handle
    conf_handle = None
    for handle, text in zip(leg1.legend_handles, leg1.get_texts(), strict=False):
        if "Confluence" in text.get_text():
            conf_handle = handle
            break
    assert conf_handle is not None
    assert conf_handle.get_marker() == "D"  # Diamond marker
    plt.close(fig1)

    # 2. plot_tree_vs_network comparative figure: both panels have complete legend by default
    G_final, stages = dgm.generate(N=25, R_rt=0.5, return_stages=True, seed=10)
    fig2, (ax_t, ax_f) = plot_tree_vs_network(stages["tree"], G_final, show_legend=True)
    assert ax_t.get_legend() is not None
    assert any("Confluence" in t.get_text() for t in ax_t.get_legend().get_texts())
    assert ax_f.get_legend() is not None
    assert any("Confluence" in t.get_text() for t in ax_f.get_legend().get_texts())
    plt.close(fig2)

    # 3. plot_stages multi-panel figure
    fig3, axes3 = plot_stages(stages)
    stage_legends = [ax.get_legend() for ax in axes3 if ax.get_legend() is not None]
    assert len(stage_legends) > 0
    assert any("Confluence" in t.get_text() for t in stage_legends[0].get_texts())
    plt.close(fig3)


def test_visualization_missing_matplotlib(monkeypatch):
    """Verify plotting functions raise an informative ImportError when matplotlib is missing."""
    import dgm.visualization as viz

    monkeypatch.setattr(viz, "_HAS_MATPLOTLIB", False)
    G = nx.DiGraph()
    G.add_edge(0, 1)

    with pytest.raises(ImportError, match="matplotlib is required"):
        viz.plot_network(G)

    with pytest.raises(ImportError, match="matplotlib is required"):
        viz.plot_tree_vs_network(G, G)

    with pytest.raises(ImportError, match="matplotlib is required"):
        viz.plot_stages({"tree": G})

