"""
Export and serialization utilities for DeltaGraphModel (DGM).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import math
import networkx as nx
import pandas as pd


def to_graphml(G: nx.DiGraph, path: Union[str, Path]) -> None:
    """Exports a DGM network graph to GraphML format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make copy to sanitize non-primitive attributes
    G_clean = nx.DiGraph()
    for n, d in G.nodes(data=True):
        clean_d = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
        G_clean.add_node(n, **clean_d)

    for u, v, d in G.edges(data=True):
        clean_d = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
        G_clean.add_edge(u, v, **clean_d)

    nx.write_graphml(G_clean, str(path))


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
            'node_type': d.get('node_type', 'internal'),
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
        props = {
            'source': u,
            'target': v,
            'feature_type': 'channel',
            'edge_type': d.get('edge_type', 'tree'),
            'width': float(d.get('width', 1.0)),
            'length': float(d.get('length', 1.0)),
            'sinuosity': float(d.get('sinuosity', 1.0)),
            'conductance': float(d.get('conductance', 1.0)),
        }
        if 'clearance_distance' in d and d['clearance_distance'] is not None:
            c_val = float(d['clearance_distance'])
            if not math.isinf(c_val) and not math.isnan(c_val):
                props['clearance_distance'] = c_val
        if 'meander_amplitude' in d and d['meander_amplitude'] is not None:
            m_val = float(d['meander_amplitude'])
            if not math.isinf(m_val) and not math.isnan(m_val):
                props['meander_amplitude'] = m_val
        if 'max_meander_amplitude' in d and d['max_meander_amplitude'] is not None:
            max_val = float(d['max_meander_amplitude'])
            if not math.isinf(max_val) and not math.isnan(max_val):
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
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    fc = {
        'type': 'FeatureCollection',
        'features': features,
        'properties': dict(G.graph)
    }

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fc, f, indent=2, default=_json_serialize_fallback)

    return fc


def metrics_to_dataframe(metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Converts a metric evaluation dictionary or list of realization metrics into a pandas DataFrame."""
    if isinstance(metrics_data, dict):
        if 'raw_realizations' in metrics_data:
            return pd.DataFrame(metrics_data['raw_realizations'])
        return pd.DataFrame([metrics_data])
    return pd.DataFrame(metrics_data)


def metrics_to_csv(metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]], path: Union[str, Path]) -> None:
    """Saves metrics to a CSV file."""
    df = metrics_to_dataframe(metrics_data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
