"""
Export and serialization utilities for DeltaGraphModel (DGM).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import math
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import networkx as nx
import numpy as np
import pandas as pd


def _sanitize_graphml_val(v: Any) -> Optional[Union[int, float, str, bool]]:
    """Sanitizes an attribute value to GraphML-compatible primitive types."""
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        val = float(v)
        return val if np.isfinite(val) else str(val)
    if isinstance(v, (str, Path)):
        return str(v)
    return None


def to_graphml(G: nx.DiGraph, path: Union[str, Path]) -> None:
    """Exports a DGM network graph to GraphML format using safe atomic writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make copy to sanitize non-primitive attributes and graph-level properties
    G_clean = nx.DiGraph()

    # Graph-level properties
    for k, v in G.graph.items():
        if k == 'stages':
            continue
        sv = _sanitize_graphml_val(v)
        if sv is not None:
            G_clean.graph[str(k)] = sv

    # Node attributes
    for n, d in G.nodes(data=True):
        clean_d = {}
        for k, v in d.items():
            sv = _sanitize_graphml_val(v)
            if sv is not None:
                clean_d[str(k)] = sv
        G_clean.add_node(n, **clean_d)

    # Edge attributes
    for u, v, d in G.edges(data=True):
        clean_d = {}
        for k, val in d.items():
            sv = _sanitize_graphml_val(val)
            if sv is not None:
                clean_d[str(k)] = sv
        G_clean.add_edge(u, v, **clean_d)

    # Atomic write via temporary file
    tmp_path = path.with_name(f".{path.name}.tmp_{uuid.uuid4().hex}")
    try:
        nx.write_graphml(G_clean, str(tmp_path))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def to_geojson(G: nx.DiGraph, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Exports a DGM graph as a GeoJSON FeatureCollection of Point (nodes) and LineString (channels).
    """
    features = []

    # Export nodes
    for n, d in G.nodes(data=True):
        x = float(d.get('x', 0.0))
        y = float(d.get('y', 0.0))
        props = {
            'id': n,
            'feature_type': 'node',
            'node_type': str(d.get('node_type', 'internal')),
            'r': float(d.get('r', 0.0)),
            'theta': float(d.get('theta', 0.0)),
            'level': int(d.get('level', 0)),
        }
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [x, y]
            },
            'properties': props
        })

    # Export edges
    for u, v, d in G.edges(data=True):
        x_u, y_u = float(G.nodes[u].get('x', 0.0)), float(G.nodes[u].get('y', 0.0))
        x_v, y_v = float(G.nodes[v].get('x', 0.0)), float(G.nodes[v].get('y', 0.0))
        coords = d.get('coordinates', [[x_u, y_u], [x_v, y_v]])
        if hasattr(coords, "tolist"):
            coords = coords.tolist()

        props = {
            'source': u,
            'target': v,
            'feature_type': 'channel',
            'edge_type': str(d.get('edge_type', 'tree')),
            'width': float(d.get('width', 1.0)),
            'length': float(d.get('length', 1.0)),
            'sinuosity': float(d.get('sinuosity', 1.0)),
            'conductance': float(d.get('conductance', 1.0)),
        }
        if 'clearance_distance' in d and d['clearance_distance'] is not None:
            c_val = float(d['clearance_distance'])
            if math.isfinite(c_val):
                props['clearance_distance'] = c_val
        if 'meander_amplitude' in d and d['meander_amplitude'] is not None:
            m_val = float(d['meander_amplitude'])
            if math.isfinite(m_val):
                props['meander_amplitude'] = m_val
        if 'max_meander_amplitude' in d and d['max_meander_amplitude'] is not None:
            max_val = float(d['max_meander_amplitude'])
            if math.isfinite(max_val):
                props['max_meander_amplitude'] = max_val

        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': coords
            },
            'properties': props
        })

    def _json_serialize_fallback(obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (set, frozenset, tuple)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items()}
        return str(obj)

    clean_graph_props = {k: v for k, v in G.graph.items() if k != 'stages'}
    fc = {
        'type': 'FeatureCollection',
        'features': features,
        'properties': clean_graph_props
    }

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp_{uuid.uuid4().hex}")
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(fc, f, indent=2, default=_json_serialize_fallback)
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    return fc


def metrics_to_dataframe(metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Converts a metric evaluation dictionary or list of realization metrics into a pandas DataFrame."""
    if isinstance(metrics_data, dict):
        if 'raw_realizations' in metrics_data:
            return pd.DataFrame(metrics_data['raw_realizations'])
        return pd.DataFrame([metrics_data])
    return pd.DataFrame(metrics_data)


def metrics_to_csv(metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]], path: Union[str, Path]) -> None:
    """Saves metrics to a CSV file using safe atomic writing."""
    df = metrics_to_dataframe(metrics_data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp_{uuid.uuid4().hex}")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
