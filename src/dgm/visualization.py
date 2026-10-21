"""
Spatial network visualization utilities for DeltaGraphModel (DGM).
Provides geomorphically accurate 2D cartographic rendering of synthetic delta
channel networks and tree backbones, including proximity-constrained sinuous
polyline centerlines, channel width hierarchies, node taxonomies, and domain boundaries.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .core.domain import PolarDomain


# Standard Geomorphic Palette (Light & Dark Themes)
THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "bg": "#ffffff",
        "panel_bg": "#f8fafc",
        "text": "#0f172a",
        "muted_text": "#64748b",
        "domain_boundary": "#94a3b8",
        "domain_fill": "#f1f5f9",
        "ocean_fill": "#e0f2fe",
        "ocean_line": "#0284c7",
        "valley_wall": "#cbd5e1",
        "edge_tree": "#0f4c81",       # Deep classic river blue
        "edge_loop": "#0891b2",       # Cyan / teal loop ties
        "edge_tidal": "#7c3aed",      # Purple tidal reconnections
        "edge_splay": "#059669",      # Emerald crevasse splays
        "edge_default": "#1e3a8a",    # Dark blue default
        "node_apex": "#dc2626",       # Crimson red
        "node_bifurc": "#16a34a",     # Emerald green
        "node_trifurc": "#0d9488",    # Teal sea-green
        "node_conf": "#d97706",       # Warm amber
        "node_complex": "#7c3aed",    # Royal purple
        "node_internal": "#64748b",   # Slate gray
        "node_outlet": "#0284c7",     # Coastal sky blue
        "node_dead_end": "#e11d48",   # Rose red
    },
    "dark": {
        "bg": "#090d16",
        "panel_bg": "#0f172a",
        "text": "#f8fafc",
        "muted_text": "#94a3b8",
        "domain_boundary": "#334155",
        "domain_fill": "#0b1220",
        "ocean_fill": "#082f49",
        "ocean_line": "#38bdf8",
        "valley_wall": "#475569",
        "edge_tree": "#38bdf8",       # Bright sky blue
        "edge_loop": "#22d3ee",       # Glowing cyan
        "edge_tidal": "#c084fc",      # Light purple
        "edge_splay": "#34d399",      # Light emerald
        "edge_default": "#60a5fa",    # Blue
        "node_apex": "#ef4444",       # Bright red
        "node_bifurc": "#4ade80",     # Neon green
        "node_trifurc": "#2dd4bf",    # Bright teal
        "node_conf": "#fbbf24",       # Amber gold
        "node_complex": "#c084fc",    # Light purple
        "node_internal": "#94a3b8",   # Slate
        "node_outlet": "#38bdf8",     # Cyan blue
        "node_dead_end": "#fb7185",   # Rose pink
    },
    "clean": {
        "bg": "none",
        "panel_bg": "none",
        "text": "#1e293b",
        "muted_text": "#64748b",
        "domain_boundary": "#94a3b8",
        "domain_fill": "none",
        "ocean_fill": "none",
        "ocean_line": "#0284c7",
        "valley_wall": "#94a3b8",
        "edge_tree": "#0f4c81",
        "edge_loop": "#0891b2",
        "edge_tidal": "#7c3aed",
        "edge_splay": "#059669",
        "edge_default": "#1e3a8a",
        "node_apex": "#dc2626",
        "node_bifurc": "#16a34a",
        "node_trifurc": "#0d9488",
        "node_conf": "#d97706",
        "node_complex": "#7c3aed",
        "node_internal": "#64748b",
        "node_outlet": "#0284c7",
        "node_dead_end": "#e11d48",
    }
}


def _transform_point(x: float, y: float, orientation: str) -> Tuple[float, float]:
    """
    Transforms 2D Cartesian coordinates (x: downstream, y: cross-stream)
    according to the specified viewing orientation.
    """
    orientation_lower = orientation.lower()
    if orientation_lower in ("downward", "seaward_down", "down"):
        # Apex at top (0, 0), flow travels downward towards -y
        return (y, -x)
    elif orientation_lower in ("upward", "seaward_up", "up"):
        # Apex at bottom (0, 0), flow travels upward towards +y
        return (y, x)
    # Cartesian default: Apex at (0, 0), flow travels rightwards towards +x
    return (x, y)


def _transform_polyline(coords: List[List[float]], orientation: str) -> np.ndarray:
    """Transforms an array of 2D polyline vertices according to orientation."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr
    orientation_lower = orientation.lower()
    if orientation_lower in ("downward", "seaward_down", "down"):
        return np.column_stack((arr[:, 1], -arr[:, 0]))
    elif orientation_lower in ("upward", "seaward_up", "up"):
        return np.column_stack((arr[:, 1], arr[:, 0]))
    return arr[:, :2]


def _infer_node_type(G: nx.DiGraph, n: Any, explicit_type: Optional[str] = None) -> str:
    """Infers geomorphic node taxonomy based on explicit metadata or topological degrees."""
    if explicit_type and explicit_type not in ("unknown", "internal", "junction"):
        return explicit_type

    in_deg = G.in_degree(n)
    out_deg = G.out_degree(n)

    if in_deg == 0 and out_deg > 0:
        return "apex"
    if out_deg == 0 and in_deg > 0:
        if explicit_type == "dead_end":
            return "dead_end"
        return "outlet"
    if out_deg >= 3 and in_deg <= 1:
        return "trifurcation"
    if out_deg >= 2 and in_deg >= 2:
        return "complex_junction"
    if out_deg == 2 and in_deg <= 1:
        return "bifurcation"
    if in_deg >= 2 and out_deg <= 1:
        return "confluence"
    return "internal"


def _extract_domain(G: nx.DiGraph, domain: Optional[PolarDomain] = None) -> Optional[PolarDomain]:
    """Retrieves or reconstructs a PolarDomain instance from graph metadata or node geometry."""
    if domain is not None:
        return domain

    # Try extracting domain parameters from graph metadata
    params = G.graph.get("params", {})
    r_shore = float(params.get("r_shore", G.graph.get("r_shore", 0.0)))
    theta_base = float(params.get("theta_base", G.graph.get("theta_base", 45.0)))
    theta_delta = float(params.get("theta_delta", G.graph.get("theta_delta", 60.0)))
    R_rt = float(params.get("R_rt", G.graph.get("R_rt", 0.5)))
    confinement = float(params.get("confinement", G.graph.get("confinement", 0.0)))
    valley_w = params.get("lateral_valley_width", G.graph.get("lateral_valley_width", None))

    if r_shore > 0:
        try:
            return PolarDomain(
                r_shore=r_shore,
                theta_base=theta_base,
                theta_delta=theta_delta,
                R_rt=R_rt,
                confinement=confinement,
                lateral_valley_width=valley_w
            )
        except Exception:
            pass

    # Infer bounds from node coordinates if present
    radii = [float(d.get("r", math.hypot(d.get("x", 0.0), d.get("y", 0.0)))) for _, d in G.nodes(data=True)]
    if radii:
        max_r = max(radii)
        if max_r > 1e-3:
            return PolarDomain(
                r_shore=max_r * 1.05,
                theta_base=theta_base,
                theta_delta=theta_delta,
                R_rt=R_rt,
                confinement=confinement,
                lateral_valley_width=valley_w
            )
    return None


def plot_network(
    G: nx.DiGraph,
    ax: Optional[plt.Axes] = None,
    curved: bool = True,
    edge_width: Union[str, float] = "width",
    min_linewidth: float = 0.8,
    max_linewidth: float = 5.5,
    width_scaling: str = "sqrt",
    edge_color: Union[str, Dict[str, str]] = "type",
    edge_alpha: float = 0.85,
    show_nodes: bool = True,
    node_size: float = 30.0,
    node_style: str = "type",
    show_domain: bool = True,
    orientation: str = "cartesian",
    theme: str = "light",
    title: Optional[str] = None,
    show_legend: bool = True,
    legend_title: Optional[str] = "Geomorphic Taxonomy",
    legend_loc: str = "upper right",
    legend_all_types: bool = True,
    legend_ncols: int = 1,
    show_stats: bool = False,
    stats_loc: str = "lower left",
    domain: Optional[PolarDomain] = None,
) -> plt.Axes:
    """
    Renders a spatial 2D map of a DeltaGraphModel (DGM) network or tree backbone.

    Depicts sinuous channel centerlines with proximity-constrained meanders,
    channel width hierarchies via Murray's law scaling, geomorphic node taxonomies,
    and polar domain shoreline boundaries.

    Parameters
    ----------
    G : nx.DiGraph
        DGM network graph with node positions ('x', 'y') and edge attributes
        ('width', 'sinuosity', optional 'coordinates').
    ax : plt.Axes, optional
        Matplotlib Axes instance. If None, a new figure and axes are created.
    curved : bool, default True
        If True, renders multi-vertex sinuous polylines from edge 'coordinates'.
        If 'coordinates' are not present, computes them on the fly from edge 'sinuosity'.
        If False, draws straight chord segments between endpoints.
    edge_width : Union[str, float], default 'width'
        Attribute name to scale line thickness by, or a constant float linewidth.
    min_linewidth : float, default 0.8
        Minimum channel line thickness in points.
    max_linewidth : float, default 5.5
        Maximum channel line thickness in points.
    width_scaling : str, default 'sqrt'
        Scaling mode for line width: 'sqrt' (hydraulic diameter), 'linear', or 'fixed'.
    edge_color : Union[str, Dict[str, str]], default 'type'
        Coloring strategy: 'type' (colors by 'edge_type'), or a valid Matplotlib color string.
    edge_alpha : float, default 0.85
        Opacity for channel polylines.
    show_nodes : bool, default True
        Whether to render nodes on the network.
    node_size : float, default 30.0
        Base marker size for nodes.
    node_style : str, default 'type'
        Strategy for styling nodes: 'type' (differentiates apex, outlets, bifurcations, confluences),
        or 'uniform'.
    show_domain : bool, default True
        Whether to draw the polar domain opening rays, shoreline boundary, and ocean fill.
    orientation : str, default 'cartesian'
        Viewing orientation: 'cartesian' (apex at origin, seaward +x),
        'downward' (apex at top, seaward -y), or 'upward' (apex at bottom, seaward +y).
    theme : str, default 'light'
        Color palette theme: 'light', 'dark', or 'clean'.
    title : Optional[str], default None
        Title string to display above the plot.
    show_legend : bool, default True
        Whether to display a geomorphic taxonomy legend.
    legend_title : Optional[str], default 'Geomorphic Taxonomy'
        Header title text for the legend box.
    legend_loc : str, default 'upper right'
        Location string for Matplotlib legend (e.g. 'upper right', 'lower right').
    legend_all_types : bool, default True
        If True, displays all defined geomorphic taxonomy node kinds and channel types
        in the legend (e.g. apex, bifurcation, trifurcation, confluence, complex junction,
        internal, outlet, dead-end). If False, only shows types actually present in G.
    legend_ncols : int, default 1
        Number of columns for legend layout.
    show_stats : bool, default False
        Whether to overlay a summary statistics card (N, E, Betti-1, mean width, sinuosity).
    stats_loc : str, default 'lower left'
        Location for the statistics box.
    domain : Optional[PolarDomain], default None
        Optional explicit domain to render boundaries from. If None, inferred automatically.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes containing the plot.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    pal = THEMES.get(theme.lower(), THEMES["light"])
    if pal["bg"] != "none":
        ax.set_facecolor(pal["panel_bg"])
        if ax.figure is not None and hasattr(ax.figure, "patch"):
            ax.figure.patch.set_facecolor(pal["bg"])

    # Empty network early exit
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty Graph", transform=ax.transAxes, ha="center", va="center", color=pal["muted_text"])
        ax.set_axis_off()
        return ax

    # Prepare working graph with coordinates
    has_node_xy = any("x" in d and "y" in d for _, d in G.nodes(data=True))
    if not has_node_xy and G.number_of_nodes() > 0:
        G_work = G.copy()
        pos = nx.spring_layout(G_work, seed=42)
        for n, (px, py) in pos.items():
            G_work.nodes[n]["x"] = float(px * 1000.0)
            G_work.nodes[n]["y"] = float(py * 1000.0)
        has_node_xy = True
    else:
        G_work = G

    has_polylines = any("coordinates" in d and len(d["coordinates"]) >= 3 for _, _, d in G_work.edges(data=True))
    if curved and not has_polylines and G_work.number_of_edges() > 0:
        if has_node_xy:
            from .core.channel_geometry import generate_channel_centerlines
            if G_work is G:
                G_work = G.copy()
            params = G_work.graph.get("params", {})
            clr = params.get("min_channel_clearance", G_work.graph.get("min_channel_clearance", 0.40))
            pts = params.get("num_polyline_points", G_work.graph.get("num_polyline_points", 21))
            bends = params.get("meander_bends", G_work.graph.get("meander_bends", 1))
            sub_seed = G_work.graph.get("seed", 42)
            try:
                generate_channel_centerlines(G_work, clearance_fraction=clr, num_points=pts, num_bends=bends, seed=sub_seed)
            except Exception:
                pass

    # Retrieve or infer PolarDomain
    resolved_domain = _extract_domain(G_work, domain)

    # 1. Render Domain Boundaries (Shoreline Arc & Opening Rays)
    if show_domain and resolved_domain is not None:
        _draw_domain(ax, resolved_domain, orientation=orientation, pal=pal)

    # 2. Compute Edge Width Scaling
    edges = list(G_work.edges(data=True))
    linewidths = []
    if edges:
        if isinstance(edge_width, (int, float)):
            linewidths = [float(edge_width)] * len(edges)
        else:
            raw_widths = np.array([float(d.get(edge_width, 1.0)) for _, _, d in edges], dtype=float)
            if width_scaling == "fixed" or np.allclose(raw_widths, raw_widths[0]):
                linewidths = [float((min_linewidth + max_linewidth) / 2.0)] * len(edges)
            else:
                if width_scaling == "sqrt":
                    scaled = np.sqrt(np.maximum(1e-3, raw_widths))
                elif width_scaling == "log":
                    scaled = np.log1p(np.maximum(1e-3, raw_widths))
                else:  # linear
                    scaled = np.maximum(1e-3, raw_widths)

                s_min, s_max = scaled.min(), scaled.max()
                if s_max > s_min:
                    norm = min_linewidth + (max_linewidth - min_linewidth) * (scaled - s_min) / (s_max - s_min)
                    linewidths = norm.tolist()
                else:
                    linewidths = [float((min_linewidth + max_linewidth) / 2.0)] * len(edges)

    # 3. Render Channels (Edges)
    type_color_map = {
        "tree": pal["edge_tree"],
        "loop": pal["edge_loop"],
        "reciprocal_loop": pal["edge_loop"],
        "tidal_loop": pal["edge_tidal"],
        "cross_tie": pal["edge_loop"],
        "crevasse_splay": pal["edge_splay"],
    }

    used_edge_types = set()
    for idx, (u, v, d) in enumerate(edges):
        u_d = G_work.nodes[u]
        v_d = G_work.nodes[v]
        x_u, y_u = float(u_d.get("x", 0.0)), float(u_d.get("y", 0.0))
        x_v, y_v = float(v_d.get("x", 0.0)), float(v_d.get("y", 0.0))

        if curved and "coordinates" in d and len(d["coordinates"]) >= 2:
            pts = _transform_polyline(d["coordinates"], orientation)
        else:
            p0 = _transform_point(x_u, y_u, orientation)
            p1 = _transform_point(x_v, y_v, orientation)
            pts = np.array([p0, p1])

        # Determine Edge Color
        etype = str(d.get("edge_type", d.get("tier", "tree")))
        used_edge_types.add(etype)

        if isinstance(edge_color, dict):
            c = edge_color.get(etype, pal["edge_default"])
        elif edge_color == "type":
            c = type_color_map.get(etype, pal["edge_default"])
        else:
            c = str(edge_color)

        lw = linewidths[idx] if idx < len(linewidths) else 1.5
        linestyle = "--" if etype in ("loop", "reciprocal_loop", "tidal_loop") else "-"

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=c,
            linewidth=lw,
            linestyle=linestyle,
            alpha=edge_alpha,
            solid_capstyle="round",
            zorder=3
        )

    # 4. Render Nodes
    node_tax_styles = {
        "apex": {"color": pal["node_apex"], "marker": "*", "size_mult": 2.7, "label": "Apex (River Source)", "zorder": 10},
        "bifurcation": {"color": pal["node_bifurc"], "marker": "o", "size_mult": 1.1, "label": "Bifurcation (Vb, 1→2)", "zorder": 8},
        "trifurcation": {"color": pal["node_trifurc"], "marker": "^", "size_mult": 1.25, "label": "Trifurcation (Vb3, 1→3+)", "zorder": 8},
        "confluence": {"color": pal["node_conf"], "marker": "D", "size_mult": 1.1, "label": "Confluence (Vc, 2→1)", "zorder": 8},
        "complex_junction": {"color": pal["node_complex"], "marker": "p", "size_mult": 1.35, "label": "Complex Junction (Multi-Way)", "zorder": 8},
        "internal": {"color": pal["node_internal"], "marker": "o", "size_mult": 0.65, "label": "Internal / Pass-Through", "zorder": 5},
        "outlet": {"color": pal["node_outlet"], "marker": "s", "size_mult": 1.35, "label": "Shoreline Outlet", "zorder": 9},
        "dead_end": {"color": pal["node_dead_end"], "marker": "X", "size_mult": 1.25, "label": "Inland Dead-End (Spur)", "zorder": 7},
    }

    used_node_types = set()
    if show_nodes:
        if node_style == "type":
            grouped_nodes: Dict[str, List[Tuple[float, float]]] = {}
            for n, d in G_work.nodes(data=True):
                nx_val = float(d.get("x", 0.0))
                ny_val = float(d.get("y", 0.0))
                px, py = _transform_point(nx_val, ny_val, orientation)
                ntype = _infer_node_type(G_work, n, d.get("node_type"))
                used_node_types.add(ntype)
                grouped_nodes.setdefault(ntype, []).append((px, py))

            for ntype, pts_list in grouped_nodes.items():
                style = node_tax_styles.get(ntype, node_tax_styles["internal"])
                s_val = node_size * style["size_mult"]
                pts_arr = np.asarray(pts_list, dtype=float)
                ax.scatter(
                    pts_arr[:, 0], pts_arr[:, 1],
                    s=s_val,
                    c=style["color"],
                    marker=style["marker"],
                    edgecolors="white" if theme == "light" else "#0f172a",
                    linewidths=0.75,
                    alpha=0.95,
                    zorder=style["zorder"]
                )
        else:
            pts_list = [
                _transform_point(float(d.get("x", 0.0)), float(d.get("y", 0.0)), orientation)
                for _, d in G_work.nodes(data=True)
            ]
            if pts_list:
                pts_arr = np.asarray(pts_list, dtype=float)
                ax.scatter(
                    pts_arr[:, 0], pts_arr[:, 1],
                    s=node_size,
                    c=pal["node_internal"],
                    marker="o",
                    edgecolors="none",
                    alpha=0.8,
                    zorder=5
                )

    # 5. Legend
    if show_legend:
        handles = []
        # Channel Handles
        if legend_all_types or "tree" in used_edge_types or not used_edge_types:
            handles.append(Line2D([0], [0], color=pal["edge_tree"], lw=2.5, label="Tree Backbone Channel"))
        if legend_all_types or any(t in used_edge_types for t in ("loop", "reciprocal_loop", "cross_tie")):
            handles.append(Line2D([0], [0], color=pal["edge_loop"], lw=2.0, ls="--", label="Emergent Loop Channel"))
        if legend_all_types or "tidal_loop" in used_edge_types:
            handles.append(Line2D([0], [0], color=pal["edge_tidal"], lw=2.0, ls="-.", label="Tidal Channel Loop"))
        if "crevasse_splay" in used_edge_types:
            handles.append(Line2D([0], [0], color=pal["edge_splay"], lw=1.5, ls=":", label="Crevasse Splay Channel"))

        # Node Handles (ordered logically from source to coast)
        if show_nodes and node_style == "type":
            ordered_node_tax = (
                "apex",
                "bifurcation",
                "trifurcation",
                "confluence",
                "complex_junction",
                "internal",
                "outlet",
                "dead_end"
            )
            for ntype in ordered_node_tax:
                if legend_all_types or ntype in used_node_types:
                    st = node_tax_styles[ntype]
                    handles.append(Line2D(
                        [0], [0],
                        marker=st["marker"],
                        color="w" if theme == "light" else "#0f172a",
                        markerfacecolor=st["color"],
                        markeredgecolor="white" if theme == "light" else "#0f172a",
                        markeredgewidth=0.75,
                        markersize=math.sqrt(node_size * st["size_mult"]) * 1.3,
                        label=st["label"],
                        ls="None"
                    ))

        # Domain boundary handles
        if show_domain and resolved_domain is not None:
            handles.append(Line2D([0], [0], color=pal["ocean_line"], lw=1.3, ls=":", label="Shoreline Boundary Arc"))
            if resolved_domain.lateral_valley_width is not None and resolved_domain.lateral_valley_width > 0:
                handles.append(Line2D([0], [0], color=pal["valley_wall"], lw=1.0, ls="-.", label="Valley Corridor Wall"))

        if handles:
            leg = ax.legend(
                handles=handles,
                loc=legend_loc,
                title=legend_title,
                fontsize=8.5,
                title_fontsize=9.2,
                ncols=legend_ncols,
                frameon=True,
                facecolor=pal["panel_bg"],
                edgecolor=pal["domain_boundary"],
                labelcolor=pal["text"],
                borderpad=0.7,
                framealpha=0.92
            )
            if leg.get_title() is not None:
                leg.get_title().set_fontweight("bold")
                leg.get_title().set_color(pal["text"])
            leg.set_zorder(10)

    # 6. Optional Statistics Overlay Badge
    if show_stats:
        betti1 = max(0, int(G_work.number_of_edges() - G_work.number_of_nodes() + nx.number_weakly_connected_components(G_work)))
        widths_all = [float(d.get("width", 0.0)) for _, _, d in G_work.edges(data=True) if "width" in d]
        sins_all = [float(d.get("sinuosity", 1.0)) for _, _, d in G_work.edges(data=True) if "sinuosity" in d]

        mean_w = np.mean(widths_all) if widths_all else 0.0
        mean_sin = np.mean(sins_all) if sins_all else 1.0

        stat_lines = [
            f"Nodes (N):      {G_work.number_of_nodes()}",
            f"Channels (E):   {G_work.number_of_edges()}",
            f"Loops (β1):     {betti1}",
            f"Mean Sinuosity: {mean_sin:.3f}",
        ]
        if mean_w > 0:
            stat_lines.append(f"Mean Width:     {mean_w:.1f} m")

        stat_text = "\n".join(stat_lines)
        loc_map = {
            "lower left": (0.03, 0.04, "left", "bottom"),
            "upper left": (0.03, 0.96, "left", "top"),
            "lower right": (0.97, 0.04, "right", "bottom"),
            "upper right": (0.97, 0.96, "right", "top"),
        }
        sx, sy, sha, sva = loc_map.get(stats_loc.lower(), (0.03, 0.04, "left", "bottom"))
        ax.text(
            sx, sy, stat_text,
            transform=ax.transAxes,
            ha=sha,
            va=sva,
            fontsize=8.0,
            family="monospace",
            color=pal["text"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor=pal["panel_bg"], edgecolor=pal["domain_boundary"], alpha=0.9),
            zorder=10
        )

    # 7. Axes Aesthetics & Aspect Ratio
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=pal["text"], pad=10)

    return ax


def _draw_domain(
    ax: plt.Axes,
    domain: PolarDomain,
    orientation: str,
    pal: Dict[str, Any]
) -> None:
    """Draws polar wedge opening rays, shoreline arc, and valley walls."""
    r_shore = domain.r_shore
    half_open = domain.half_open

    # 1. Opening Rays (Apex to Shoreline corners)
    ray_top = [
        _transform_point(0.0, 0.0, orientation),
        _transform_point(r_shore * math.cos(half_open), r_shore * math.sin(half_open), orientation)
    ]
    ray_bot = [
        _transform_point(0.0, 0.0, orientation),
        _transform_point(r_shore * math.cos(-half_open), r_shore * math.sin(-half_open), orientation)
    ]

    ax.plot([ray_top[0][0], ray_top[1][0]], [ray_top[0][1], ray_top[1][1]],
            color=pal["domain_boundary"], lw=0.9, ls="--", alpha=0.6, zorder=1)
    ax.plot([ray_bot[0][0], ray_bot[1][0]], [ray_bot[0][1], ray_bot[1][1]],
            color=pal["domain_boundary"], lw=0.9, ls="--", alpha=0.6, zorder=1)

    # 2. Shoreline Arc
    t_vals = np.linspace(-half_open, half_open, 150)
    arc_x = r_shore * np.cos(t_vals)
    arc_y = r_shore * np.sin(t_vals)
    arc_pts = np.array([_transform_point(x, y, orientation) for x, y in zip(arc_x, arc_y)])

    ax.plot(
        arc_pts[:, 0], arc_pts[:, 1],
        color=pal["ocean_line"],
        lw=1.3,
        ls=":",
        alpha=0.7,
        zorder=1
    )

    # 3. Lateral Valley Walls (if corridor confinement is active)
    if domain.lateral_valley_width is not None and domain.lateral_valley_width > 0:
        hw = domain.lateral_valley_width / 2.0
        if hw < r_shore:
            x_wall_end = math.sqrt(r_shore**2 - hw**2)
        else:
            x_wall_end = r_shore
        x_ray_end = r_shore * math.cos(half_open)
        x_end = max(x_ray_end, x_wall_end)

        wall_top_0 = _transform_point(0.0, hw, orientation)
        wall_top_1 = _transform_point(x_end, hw, orientation)
        wall_bot_0 = _transform_point(0.0, -hw, orientation)
        wall_bot_1 = _transform_point(x_end, -hw, orientation)

        ax.plot([wall_top_0[0], wall_top_1[0]], [wall_top_0[1], wall_top_1[1]],
                color=pal["valley_wall"], lw=1.0, ls="-.", alpha=0.5, zorder=1)
        ax.plot([wall_bot_0[0], wall_bot_1[0]], [wall_bot_0[1], wall_bot_1[1]],
                color=pal["valley_wall"], lw=1.0, ls="-.", alpha=0.5, zorder=1)


def plot_tree_vs_network(
    G_tree: nx.DiGraph,
    G_final: nx.DiGraph,
    curved: bool = True,
    orientation: str = "cartesian",
    theme: str = "light",
    figsize: Tuple[float, float] = (16, 7),
    dpi: int = 200,
    title: Optional[str] = None,
    show_stats: bool = True,
    show_legend: bool = True,
    legend_panel: str = "both",
    legend_all_types: bool = True,
    **kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Renders a side-by-side comparative figure of the Stage 1 Polar Tree Backbone
    versus the Stage 6 Final Hydraulic Delta Network.

    Parameters
    ----------
    G_tree : nx.DiGraph
        Initial bifurcating tree backbone before loop closure and pruning.
    G_final : nx.DiGraph
        Final simulated delta network with emergent loops, hydraulics, and sinuosity.
    curved : bool, default True
        Whether to depict sinuous polyline channels.
    orientation : str, default 'cartesian'
        Coordinate orientation ('cartesian', 'downward', 'upward').
    theme : str, default 'light'
        Color palette theme ('light', 'dark', 'clean').
    figsize : Tuple[float, float], default (16, 7)
        Matplotlib figure dimensions (width, height) in inches.
    dpi : int, default 200
        Figure resolution.
    title : Optional[str], default None
        Overall figure title.
    show_stats : bool, default True
        Whether to display summary stats cards in both panels.
    show_legend : bool, default True
        Whether to show the legend.
    legend_panel : str, default 'both'
        Which panel(s) should render the legend: 'both', 'final', or 'tree'.
    legend_all_types : bool, default True
        Whether the legend should list all standard node and edge kinds in the taxonomy.
    **kwargs
        Additional arguments forwarded to `plot_network`.

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Figure and pair of (ax_tree, ax_final) axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    pal = THEMES.get(theme.lower(), THEMES["light"])
    if pal["bg"] != "none":
        fig.patch.set_facecolor(pal["bg"])

    # Panel 1: Tree Backbone
    show_leg_1 = show_legend and (legend_panel.lower() in ("tree", "both"))
    plot_network(
        G_tree,
        ax=ax1,
        curved=curved,
        orientation=orientation,
        theme=theme,
        title="Stage 1: Polar Tree Backbone (Dendritic Skeleton)",
        show_stats=show_stats,
        show_legend=show_leg_1,
        legend_all_types=legend_all_types,
        **kwargs
    )

    # Panel 2: Final Network
    show_leg_2 = show_legend and (legend_panel.lower() in ("final", "both"))
    plot_network(
        G_final,
        ax=ax2,
        curved=curved,
        orientation=orientation,
        theme=theme,
        title="Final Network: Hydraulics, Loops & Sinuous Channels",
        show_stats=show_stats,
        show_legend=show_leg_2,
        legend_all_types=legend_all_types,
        **kwargs
    )

    # Synchronize limits between panels for seamless visual comparison
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=pal["text"], y=0.98)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_stages(
    stages: Dict[str, nx.DiGraph],
    curved: bool = True,
    orientation: str = "cartesian",
    theme: str = "light",
    figsize: Tuple[float, float] = (22, 6),
    dpi: int = 200,
    title: Optional[str] = "DGM Process-Emergent Generative Evolution",
    legend_stage: str = "first",
    legend_all_types: bool = True,
    **kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Renders progressive multi-panel evolution across simulation stages:
    (Tree Backbone -> Looped Network -> Pruned Network -> Final Hydraulic Network).

    Parameters
    ----------
    stages : Dict[str, nx.DiGraph]
        Dictionary mapping stage names (e.g. 'tree', 'looped', 'pruned', 'final')
        to their respective graph objects.
    curved : bool, default True
        Whether to depict sinuous channel centerlines.
    orientation : str, default 'cartesian'
        Coordinate orientation.
    theme : str, default 'light'
        Color theme ('light', 'dark', 'clean').
    figsize : Tuple[float, float], default (22, 6)
        Figure dimensions.
    dpi : int, default 200
        Figure resolution.
    title : Optional[str]
        Overall figure title.
    legend_stage : str, default 'first'
        Which stage panel to show the legend on: 'first', 'final', or 'all'.
    legend_all_types : bool, default True
        Whether the legend should show all taxonomy node and channel types.
    **kwargs
        Additional arguments forwarded to `plot_network`.

    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        Figure and list of Axes objects.
    """
    stage_order = [k for k in ["tree", "looped", "pruned", "final"] if k in stages]
    if not stage_order:
        stage_order = list(stages.keys())

    n_stages = len(stage_order)
    fig, axes = plt.subplots(1, n_stages, figsize=figsize, dpi=dpi)
    if n_stages == 1:
        axes = [axes]

    pal = THEMES.get(theme.lower(), THEMES["light"])
    if pal["bg"] != "none":
        fig.patch.set_facecolor(pal["bg"])

    stage_titles = {
        "tree": "1. Polar Tree Backbone",
        "looped": "2. + Emergent Loops",
        "pruned": "3. Simplified & Pruned",
        "final": "4. Murray Hydraulics & Meanders",
    }

    for idx, key in enumerate(stage_order):
        ax = axes[idx]
        g = stages[key]
        if legend_stage == "all":
            show_l = True
        elif legend_stage == "final":
            show_l = (idx == n_stages - 1)
        else:
            show_l = (idx == 0)

        plot_network(
            g,
            ax=ax,
            curved=curved,
            orientation=orientation,
            theme=theme,
            title=stage_titles.get(key, f"Stage: {key.capitalize()}"),
            show_stats=True,
            show_legend=show_l,
            legend_all_types=legend_all_types,
            **kwargs
        )

    # Synchronize limits across all stages
    x_mins = [ax.get_xlim()[0] for ax in axes]
    x_maxs = [ax.get_xlim()[1] for ax in axes]
    y_mins = [ax.get_ylim()[0] for ax in axes]
    y_maxs = [ax.get_ylim()[1] for ax in axes]

    x_glob_min, x_glob_max = min(x_mins), max(x_maxs)
    y_glob_min, y_glob_max = min(y_mins), max(y_maxs)

    if x_glob_min == x_glob_max:
        x_glob_min -= 1.0
        x_glob_max += 1.0
    if y_glob_min == y_glob_max:
        y_glob_min -= 1.0
        y_glob_max += 1.0

    for ax in axes:
        ax.set_xlim(x_glob_min, x_glob_max)
        ax.set_ylim(y_glob_min, y_glob_max)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=pal["text"], y=0.98)

    plt.tight_layout()
    return fig, axes


# Convenience Alias
plot_delta = plot_network
