"""
Empirical datasets and DIDNT benchmark network loader for DGM.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from ..metrics import MetricsBattery

CANONICAL_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "canonical_empirical_deltas.json"


def _resolve_didnt_dir(dataset_dir: str | Path | None = None) -> Path:
    """Resolves path to DIDNT dataset directory containing NetworkNodesAndLinks."""
    if dataset_dir is not None:
        base_path = Path(dataset_dir)
    else:
        # Check standard standalone locations
        env_dir = os.environ.get("DGM_DIDNT_DIR") or os.environ.get("DIDNT_DATASET_DIR")
        if env_dir:
            base_path = Path(env_dir)
        else:
            cwd = Path.cwd()
            repo_root = Path(__file__).resolve().parents[3]
            candidates = [
                cwd / "datasets" / "DIDNT",
                cwd / "DIDNT",
                Path.home() / ".dgm" / "datasets" / "DIDNT",
                repo_root / "datasets" / "DIDNT",
            ]
            base_path = None
            for cand in candidates:
                if cand.exists():
                    base_path = cand
                    break
            if base_path is None:
                base_path = Path("datasets/DIDNT")

    if (base_path / "NetworkNodesAndLinks").exists():
        return base_path
    elif (base_path / "DIDNT" / "NetworkNodesAndLinks").exists():
        return base_path / "DIDNT"
    elif (base_path / "DIDNT" / "DIDNT" / "NetworkNodesAndLinks").exists():
        return base_path / "DIDNT" / "DIDNT"
    return base_path



class DIDNTLoader:
    """
    Loader for the empirical DIDNT dataset (Hariharan et al., 2022).
    Parses GeoJSON node/link networks into NetworkX DiGraph objects.
    """

    def __init__(self, dataset_dir: str | Path | None = None):
        self.root_dir = _resolve_didnt_dir(dataset_dir)
        self.nnl_dir = self.root_dir / "NetworkNodesAndLinks"
        self.raw_dir = self.root_dir / "RawData"

    def is_available(self) -> bool:
        """Returns True if the DIDNT dataset files are present on disk."""
        return self.nnl_dir.exists()

    def list_available_deltas(self) -> list[str]:
        """Returns a sorted list of all delta network names available in the dataset."""
        if not self.nnl_dir.exists():
            return []
        nodes_files = list(self.nnl_dir.glob("*_nodes.json")) + list(self.nnl_dir.glob("*_nodes.geojson"))
        return sorted({f.stem.replace("_nodes", "") for f in nodes_files})

    def load_graph(self, delta_name: str) -> nx.DiGraph:
        """
        Loads the directed channel network graph for a given delta.
        """
        if not self.nnl_dir.exists():
            raise FileNotFoundError(
                f"DIDNT dataset directory not found at {self.nnl_dir}. "
                "Please ensure the dataset is downloaded or extracted."
            )

        nodes_file = self.nnl_dir / f"{delta_name}_nodes.json"
        links_file = self.nnl_dir / f"{delta_name}_links.json"

        if not nodes_file.exists():
            nodes_file = self.nnl_dir / f"{delta_name}_nodes.geojson"
            links_file = self.nnl_dir / f"{delta_name}_links.geojson"

        if not nodes_file.exists():
            matches = list(self.nnl_dir.glob(f"*{delta_name}*_nodes.*json*"))
            if matches:
                nodes_file = matches[0]
                delta_name = nodes_file.stem.replace("_nodes", "")
                ext = nodes_file.suffix
                links_file = self.nnl_dir / f"{delta_name}_links{ext}"
            else:
                raise FileNotFoundError(f"Nodes file not found for delta '{delta_name}' in {self.nnl_dir}")

        if not links_file.exists():
            link_matches = list(self.nnl_dir.glob(f"*{delta_name}*_links.*json*"))
            if link_matches:
                links_file = link_matches[0]
            else:
                raise FileNotFoundError(f"Links file not found for delta '{delta_name}' in {self.nnl_dir}")

        with open(nodes_file, encoding='utf-8') as f:
            nodes_geojson = json.load(f)

        with open(links_file, encoding='utf-8') as f:
            links_geojson = json.load(f)

        G = nx.DiGraph()
        G.graph['name'] = delta_name

        apex_coord = None
        for feat in nodes_geojson.get('features', []):
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})
            coords = geom.get('coordinates', [0.0, 0.0])
            x = float(coords[0]) if len(coords) > 0 else 0.0
            y = float(coords[1]) if len(coords) > 1 else 0.0

            nid = props.get('id', props.get('node_id', feat.get('id')))
            if nid is None:
                continue
            try:
                nid = int(nid)
            except (ValueError, TypeError):
                nid = str(nid)

            dist_apex = props.get('distance_from_apex')
            if dist_apex is not None:
                try:
                    dist_apex = float(dist_apex)
                except (ValueError, TypeError):
                    dist_apex = None

            is_inlet = (dist_apex == 0.0) or bool(props.get('inlet', False) or props.get('is_inlet', False) or props.get('type') == 'inlet')
            is_outlet = bool(props.get('outlet', False) or props.get('is_outlet', False) or props.get('type') == 'outlet')

            node_type = 'apex' if is_inlet else ('outlet' if is_outlet else 'junction')

            G.add_node(
                nid,
                x=x,
                y=y,
                distance_from_apex=dist_apex,
                is_apex=is_inlet,
                is_outlet=is_outlet,
                node_type=node_type,
                properties=props
            )

        # Load links
        for feat in links_geojson.get('features', []):
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})

            src = props.get('start_node', props.get('source', props.get('from')))
            tgt = props.get('end_node', props.get('target', props.get('to')))

            if (src is None or tgt is None) and 'conn' in props:
                conn_val = str(props['conn'])
                parts = [p.strip() for p in conn_val.split(',')]
                if len(parts) >= 2:
                    src, tgt = parts[0], parts[1]

            if src is None or tgt is None:
                continue
            try:
                src = int(src)
            except (ValueError, TypeError):
                src = str(src)
            try:
                tgt = int(tgt)
            except (ValueError, TypeError):
                tgt = str(tgt)

            if not G.has_node(src) or not G.has_node(tgt):
                continue

            raw_w = props.get('avgwidth', props.get('wid_med', props.get('width', props.get('wid', 10.0))))
            width = float(raw_w) if raw_w is not None else 10.0
            width = max(0.1, width)

            raw_len = props.get('length', props.get('len', 100.0))
            length = float(raw_len) if raw_len is not None else 100.0
            length = max(1.0, length)

            conductance = float(props.get('conductance', (width ** 1.65) / (length ** 0.50)))

            G.add_edge(
                src,
                tgt,
                width=width,
                length=length,
                conductance=max(1e-6, conductance),
                sinuosity=float(props.get('sinuosity', 1.0)),
                edge_type='tree',
                properties=props
            )

        # Remove completely isolated nodes (degree 0 artifacts in GeoJSON)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        if isolated:
            G.remove_nodes_from(isolated)

        # Identify apex node (must have out_degree > 0 to feed flow into network)
        apex_candidates = [
            n for n in G.nodes()
            if G.out_degree(n) > 0 and (
                G.nodes[n].get('is_apex') or G.in_degree(n) == 0
            )
        ]
        if not apex_candidates:
            apex_candidates = [n for n in G.nodes() if G.out_degree(n) > 0]

        apex_node = None
        if apex_candidates:
            apex_node = min(
                apex_candidates,
                key=lambda n: G.nodes[n].get("distance_from_apex")
                if G.nodes[n].get("distance_from_apex") is not None
                else 999999.0
            )
        elif len(G) > 0:
            apex_node = next(iter(G.nodes()))

        if apex_node is not None and apex_node in G.nodes():
            for n in G.nodes():
                G.nodes[n]['is_apex'] = (n == apex_node)
            G.nodes[apex_node]['node_type'] = 'apex'
            apex_coord = (G.nodes[apex_node]['x'], G.nodes[apex_node]['y'])
        else:
            apex_coord = (0.0, 0.0)

        # Compute polar coordinates relative to apex
        for _, d in G.nodes(data=True):
            dx = float(d.get('x', 0.0)) - apex_coord[0]
            dy = float(d.get('y', 0.0)) - apex_coord[1]
            d['r'] = float(math.hypot(dx, dy))
            d['theta'] = float(math.atan2(dy, dx))

        for n in G.nodes():
            if G.out_degree(n) == 0 and n != apex_node:
                G.nodes[n]["node_type"] = "outlet"
                G.nodes[n]["is_outlet"] = True
            else:
                if n != apex_node:
                    G.nodes[n]["node_type"] = "junction"
                    G.nodes[n]["is_outlet"] = False
                else:
                    G.nodes[n]["is_outlet"] = False

        return G


@dataclass
class EmpiricalDelta:
    name: str
    nodes: int
    edges: int
    loops: int
    alpha: float
    betti_1_norm: float
    vc_vb_ratio: float
    topological_slope: float
    horton_inversion: float
    recombination_factor: float
    nER_star: float
    eBI: float
    log_eBI: float
    algebraic_connectivity: float
    energy_dissipation: float
    log_energy_dissipation: float
    energy_entropy_ratio: float
    global_vulnerability: float
    outlet_flux_entropy: float
    width_cv: float
    radial_sinuosity: float
    angular_dispersion: float
    loop_length_anisotropy: float
    num_outlets: int
    log_num_outlets: float
    resistance_distance: float
    link_sharing_index: float
    forcing_type: str
    estimated_R_rt: float
    source: str = "DIDNT Dataset (Hariharan et al., 2022)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DELTA_FORCING_REGIMES: dict[str, dict[str, Any]] = {
    'Amazon': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.25},
    'Colville': {'forcing_type': 'arctic', 'estimated_R_rt': 0.40, 'physical_R_rt': 0.08, 'anastomosis_P0': 0.12},
    'Danube': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.10},
    'Dnieper': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.10},
    'Fly': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.85},
    'GBMD': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.80},
    'Hay': {'forcing_type': 'lacustrine', 'estimated_R_rt': 0.05},
    'Irrawaddy': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.50},
    'Krishna': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.50},
    'Lena': {'forcing_type': 'arctic', 'estimated_R_rt': 0.45, 'physical_R_rt': 0.08, 'anastomosis_P0': 0.14},
    'Mackenzie': {'forcing_type': 'arctic', 'estimated_R_rt': 0.40, 'physical_R_rt': 0.08, 'anastomosis_P0': 0.12},
    'Mahakam': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.60},
    'Mahanadi': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.55},
    'Mekong': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.65},
    'Mississippi': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.05},
    'Mossy': {'forcing_type': 'lacustrine', 'estimated_R_rt': 0.05},
    'Niger': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.90},
    'Orinoco': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.15},
    'Parana': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.15},
    'Pearl': {'forcing_type': 'tide_influenced', 'estimated_R_rt': 0.70},
    'Po': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.15},
    'Red': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.20},
    'Rioni': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.10},
    'Selenga': {'forcing_type': 'lacustrine', 'estimated_R_rt': 0.05},
    'Wairau': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.10},
    'Wax Lake': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.10, 'didnt_name': 'Wax'},
    'Yangtze': {'forcing_type': 'river_dominated', 'estimated_R_rt': 0.25},
    'Yukon': {'forcing_type': 'arctic', 'estimated_R_rt': 0.35, 'physical_R_rt': 0.08, 'anastomosis_P0': 0.10},
}


def load_bundled_canonical_deltas() -> list[EmpiricalDelta]:
    """Loads bundled canonical empirical deltas from package JSON data with fallback search paths."""
    candidates = [
        CANONICAL_DATA_PATH,
        Path(__file__).resolve().parent.parent / "data" / "canonical_empirical_deltas.json",
        Path(__file__).resolve().parent / "data" / "canonical_empirical_deltas.json",
        Path.cwd() / "src" / "dgm" / "data" / "canonical_empirical_deltas.json",
        Path.cwd() / "dgm" / "src" / "dgm" / "data" / "canonical_empirical_deltas.json",
    ]
    target_path = None
    for cand in candidates:
        if cand.exists():
            target_path = cand
            break

    if target_path is None:
        return []

    try:
        with open(target_path, encoding="utf-8") as f:
            raw_list = json.load(f)
        return [EmpiricalDelta(**item) for item in raw_list]
    except Exception:
        return []


_CACHED_EMPIRICAL_DELTAS: list[EmpiricalDelta] | None = None


def get_empirical_deltas(dataset_dir: str | Path | None = None, force_reload: bool = False) -> list[EmpiricalDelta]:
    """
    Loads empirical deltas from DIDNT dataset and evaluates their metrics.
    If raw DIDNT GeoJSON files are not present on disk, falls back gracefully to the
    bundled canonical pre-computed metrics dataset.
    """
    global _CACHED_EMPIRICAL_DELTAS
    if _CACHED_EMPIRICAL_DELTAS is not None and not force_reload and dataset_dir is None:
        return _CACHED_EMPIRICAL_DELTAS

    # If dataset_dir is None and not force_reload, prefer fast bundled canonical deltas
    if dataset_dir is None and not force_reload:
        bundled = load_bundled_canonical_deltas()
        if bundled:
            _CACHED_EMPIRICAL_DELTAS = bundled
            return bundled

    loader = DIDNTLoader(dataset_dir)
    if not loader.is_available():
        bundled = load_bundled_canonical_deltas()
        if dataset_dir is None and bundled:
            _CACHED_EMPIRICAL_DELTAS = bundled
        return bundled

    battery = MetricsBattery()
    deltas = []

    for name, info in DELTA_FORCING_REGIMES.items():
        didnt_name = info.get('didnt_name', name)
        try:
            G = loader.load_graph(didnt_name)
            m = battery.evaluate_all(G)

            emp = EmpiricalDelta(
                name=name,
                nodes=int(m['num_nodes']),
                edges=int(m['num_edges']),
                loops=int(m['betti_1']),
                alpha=float(m['alpha_index']),
                betti_1_norm=float(m['betti_1_norm']),
                vc_vb_ratio=float(m['ratio_vc_vb']),
                topological_slope=float(m['topological_slope']),
                horton_inversion=float(m['horton_inversion_index']),
                recombination_factor=float(m.get('recombination_factor', m['ratio_vc_vb'])),
                nER_star=float(m['nER_star']),
                eBI=float(m['eBI']),
                log_eBI=float(m.get('log_eBI', np.log1p(m['eBI']))),
                algebraic_connectivity=float(m['algebraic_connectivity']),
                energy_dissipation=float(m['energy_dissipation']),
                log_energy_dissipation=float(m.get('log_energy_dissipation', np.log1p(m['energy_dissipation']))),
                energy_entropy_ratio=float(m.get('energy_entropy_ratio', 0.0)),
                global_vulnerability=float(m.get('global_vulnerability', 0.5)),
                outlet_flux_entropy=float(m.get('outlet_flux_entropy', 0.0)),
                width_cv=float(m['width_cv']),
                radial_sinuosity=float(m.get('radial_sinuosity', 1.0)),
                angular_dispersion=float(m.get('angular_dispersion', 0.5)),
                loop_length_anisotropy=float(m.get('loop_length_anisotropy', 1.0)),
                num_outlets=int(m['num_outlets']),
                log_num_outlets=float(m.get('log_num_outlets', np.log1p(m['num_outlets']))),
                resistance_distance=float(m['resistance_distance']),
                link_sharing_index=float(m['link_sharing_index']),
                forcing_type=info['forcing_type'],
                estimated_R_rt=float(info['estimated_R_rt']),
                source="DIDNT Dataset (Hariharan et al., 2022)"
            )
            deltas.append(emp)
        except Exception:
            continue

    if dataset_dir is None:
        _CACHED_EMPIRICAL_DELTAS = deltas
    return deltas


def get_empirical_deltas_dict(dataset_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Export helper providing direct dictionary access to empirical data for deltas."""
    return {d.name: d.to_dict() for d in get_empirical_deltas(dataset_dir=dataset_dir)}


class _LazyEmpiricalDeltasDict(dict):
    """
    Lazy dictionary proxy for CANONICAL_EMPIRICAL_DELTAS that defers disk I/O
    until the dictionary is first accessed while remaining a full dict subclass.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            super().update(get_empirical_deltas_dict())
            self._loaded = True

    def __getitem__(self, key: str) -> dict[str, Any]:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        self._ensure_loaded()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._ensure_loaded()
        super().__delitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def __bool__(self) -> bool:
        self._ensure_loaded()
        return bool(super().__len__())

    def __repr__(self) -> str:
        self._ensure_loaded()
        return super().__repr__()

    def __eq__(self, other: object) -> bool:
        self._ensure_loaded()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        self._ensure_loaded()
        return super().__ne__(other)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().get(key, default)

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def copy(self) -> dict[str, dict[str, Any]]:
        self._ensure_loaded()
        return super().copy()

    def pop(self, key: str, *args: Any) -> Any:
        self._ensure_loaded()
        return super().pop(key, *args)

    def popitem(self) -> Any:
        self._ensure_loaded()
        return super().popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().setdefault(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_loaded()
        super().update(*args, **kwargs)

    def clear(self) -> None:
        self._loaded = True
        super().clear()


# Backwards compatibility aliases
EmpiricalDeltaV2 = EmpiricalDelta
get_empirical_deltas_v2 = get_empirical_deltas
CANONICAL_EMPIRICAL_DELTAS: dict[str, dict[str, Any]] = _LazyEmpiricalDeltasDict()
