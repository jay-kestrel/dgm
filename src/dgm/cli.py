"""
Command Line Interface (CLI) for DeltaGraphModel (DGM).
"""

import argparse
import sys
from pathlib import Path

from .config import DGMConfig
from .generator import DeltaGenerator, EnsembleGenerator
from .io.export import to_graphml, to_geojson, metrics_to_csv


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="dgm",
        description="DeltaGraphModel (DGM): Process-emergent generative delta channel network simulator."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate a single synthetic delta graph realization")
    gen_parser.add_argument("-N", "--nodes", type=int, default=50, help="Target number of nodes/junctions (default: 50)")
    gen_parser.add_argument("-R", "--r-rt", type=float, default=0.5, help="River-to-Tidal forcing ratio in [0, 1] (default: 0.5)")
    gen_parser.add_argument("-C", "--confinement", type=float, default=None, help="Lateral valley confinement in [0, 0.95] (default: 0.0)")
    gen_parser.add_argument("-W", "--lateral-valley-width", type=float, default=None, help="Explicit maximum lateral valley corridor width in meters (default: unconstrained)")
    gen_parser.add_argument("--coastal-node-decay", type=float, default=None, help="Shoreward decay rate of radial node spacing in [0, 0.8]")
    gen_parser.add_argument("--coastal-sinuosity-decay", type=float, default=None, help="Shoreward decay rate of channel sinuosity excess in [0, 0.9]")
    gen_parser.add_argument("--curved", "--enable-curved-centerlines", dest="curved", action="store_true", default=None, help="Generate curved polyline channel centerlines")
    gen_parser.add_argument("--min-channel-clearance", type=float, default=None, help="Channel clearance fraction for meander amplitude in [0.05, 1.0]")
    gen_parser.add_argument("-m", "--mechanism", choices=["potential_field", "ner_optimality"], default="potential_field", help="Loop reconnection engine")
    gen_parser.add_argument("-c", "--config", type=str, default=None, help="Path to custom YAML configuration file")
    gen_parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    gen_parser.add_argument("-o", "--output", type=str, default=None, help="Output file path (.graphml, .geojson, or .json)")

    # Command: ensemble
    ens_parser = subparsers.add_parser("ensemble", help="Sample an ensemble cloud of delta realizations")
    ens_parser.add_argument("-N", "--nodes", type=int, default=50, help="Target number of nodes/junctions (default: 50)")
    ens_parser.add_argument("-R", "--r-rt", type=float, default=0.5, help="River-to-Tidal forcing ratio in [0, 1] (default: 0.5)")
    ens_parser.add_argument("-C", "--confinement", type=float, default=None, help="Lateral valley confinement in [0, 0.95] (default: 0.0)")
    ens_parser.add_argument("-W", "--lateral-valley-width", type=float, default=None, help="Explicit maximum lateral valley corridor width in meters (default: unconstrained)")
    ens_parser.add_argument("--coastal-node-decay", type=float, default=None, help="Shoreward decay rate of radial node spacing in [0, 0.8]")
    ens_parser.add_argument("--coastal-sinuosity-decay", type=float, default=None, help="Shoreward decay rate of channel sinuosity excess in [0, 0.9]")
    ens_parser.add_argument("--curved", "--enable-curved-centerlines", dest="curved", action="store_true", default=None, help="Generate curved polyline channel centerlines")
    ens_parser.add_argument("--min-channel-clearance", type=float, default=None, help="Channel clearance fraction for meander amplitude in [0.05, 1.0]")
    ens_parser.add_argument("-n", "--num-realizations", type=int, default=30, help="Number of realizations to sample (default: 30)")
    ens_parser.add_argument("-m", "--mechanism", choices=["potential_field", "ner_optimality"], default="potential_field", help="Loop reconnection engine")
    ens_parser.add_argument("-c", "--config", type=str, default=None, help="Path to custom YAML configuration file")
    ens_parser.add_argument("--sample-parameters", action="store_true", default=True, help="Sample parameters from YAML distributions")
    ens_parser.add_argument("-s", "--seed", type=int, default=42, help="Base random seed (default: 42)")
    ens_parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path for ensemble summary metrics")

    parsed = parser.parse_args(args)

    if parsed.command == "generate":
        if parsed.config:
            cfg = DGMConfig.from_yaml(parsed.config, N=parsed.nodes, R_rt=parsed.r_rt, loop_mechanism=parsed.mechanism, seed=parsed.seed)
        else:
            cfg = DGMConfig.load_default()
            cfg.N = parsed.nodes
            cfg.R_rt = parsed.r_rt
            cfg.loop_mechanism = parsed.mechanism
            cfg.seed = parsed.seed

        if parsed.confinement is not None:
            cfg.confinement = parsed.confinement
        if parsed.lateral_valley_width is not None:
            cfg.lateral_valley_width = parsed.lateral_valley_width
        if parsed.coastal_node_decay is not None:
            cfg.coastal_node_decay = parsed.coastal_node_decay
        if parsed.coastal_sinuosity_decay is not None:
            cfg.coastal_sinuosity_decay = parsed.coastal_sinuosity_decay
        if parsed.curved is not None:
            cfg.enable_curved_centerlines = parsed.curved
        if parsed.min_channel_clearance is not None:
            cfg.min_channel_clearance = parsed.min_channel_clearance

        gen = DeltaGenerator(cfg)
        G = gen.generate()
        metrics = gen.evaluate_metrics(G)

        print(f"=== Generated DGM Realization ===")
        print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Loops (Betti-1): {metrics.get('betti_1', 0)}")
        print(f"Alpha Index (α): {metrics.get('alpha_index', 0.0):.4f}")
        print(f"Normalized Entropy Rate (nER*): {metrics.get('nER_star', 0.0):.4f}")
        print(f"Entropic Braiding Index (eBI): {metrics.get('eBI', 0.0):.4f}")

        if parsed.output:
            out_path = Path(parsed.output)
            if out_path.suffix.lower() == ".graphml":
                to_graphml(G, out_path)
            elif out_path.suffix.lower() in (".geojson", ".json"):
                to_geojson(G, out_path)
            else:
                to_graphml(G, out_path.with_suffix(".graphml"))
            print(f"Saved graph to {parsed.output}")

    elif parsed.command == "ensemble":
        if parsed.config:
            cfg = DGMConfig.from_yaml(parsed.config, N=parsed.nodes, R_rt=parsed.r_rt, loop_mechanism=parsed.mechanism)
        else:
            cfg = DGMConfig.load_default()
            cfg.N = parsed.nodes
            cfg.R_rt = parsed.r_rt
            cfg.loop_mechanism = parsed.mechanism

        if parsed.confinement is not None:
            cfg.confinement = parsed.confinement
            cfg.distributions.pop("confinement", None)
        if parsed.lateral_valley_width is not None:
            cfg.lateral_valley_width = parsed.lateral_valley_width
            cfg.distributions.pop("lateral_valley_width", None)
        if parsed.coastal_node_decay is not None:
            cfg.coastal_node_decay = parsed.coastal_node_decay
            cfg.distributions.pop("coastal_node_decay", None)
        if parsed.coastal_sinuosity_decay is not None:
            cfg.coastal_sinuosity_decay = parsed.coastal_sinuosity_decay
            cfg.distributions.pop("coastal_sinuosity_decay", None)
        if parsed.curved is not None:
            cfg.enable_curved_centerlines = parsed.curved
        if parsed.min_channel_clearance is not None:
            cfg.min_channel_clearance = parsed.min_channel_clearance
            cfg.distributions.pop("min_channel_clearance", None)

        cfg.sample_parameters = parsed.sample_parameters

        ens = EnsembleGenerator(
            N=parsed.nodes,
            R_rt=parsed.r_rt,
            num_realizations=parsed.num_realizations,
            loop_mechanism=parsed.mechanism,
            sample_parameters=parsed.sample_parameters,
            base_seed=parsed.seed,
            config=cfg
        )
        print(f"Sampling {parsed.num_realizations} realizations for N={parsed.nodes}, R_rt={parsed.r_rt}...")
        summary = ens.evaluate_ensemble()

        alpha_mean = summary['metrics'].get('alpha_index', {}).get('mean', 0.0)
        ner_mean = summary['metrics'].get('nER_star', {}).get('mean', 0.0)
        ebi_mean = summary['metrics'].get('eBI', {}).get('mean', 0.0)

        print(f"=== Ensemble Summary ({parsed.num_realizations} Realizations) ===")
        print(f"Alpha Index (α): mean = {alpha_mean:.4f}")
        print(f"Normalized Entropy Rate (nER*): mean = {ner_mean:.4f}")
        print(f"Entropic Braiding Index (eBI): mean = {ebi_mean:.4f}")

        if parsed.output:
            metrics_to_csv(summary, parsed.output)
            print(f"Saved ensemble metrics to {parsed.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
