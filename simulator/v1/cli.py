"""Command-line entry point for the MoE V1 simulator."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import V1ConfigError, load_v1_config
from .outputs import write_v1_outputs
from .plots import write_v1_plots
from .simulator import V1Simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MoE V1 cache/prefetch simulator")
    parser.add_argument("config", type=Path, help="V1 YAML configuration path")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_v1_config(args.config)
        result = V1Simulator(config).run()
        paths = write_v1_outputs(result, args.output_dir)
        plot_paths = write_v1_plots(result, config, args.output_dir)
    except V1ConfigError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    summary = result.summary
    print("MoE V1 simulation summary")
    for field in (
        "status", "total_cycles", "initial_kv_bytes", "final_kv_bytes",
        "cache_hit_rate", "observed_prediction_accuracy",
        "useful_prefetch_bytes", "wasted_prefetch_bytes",
        "demand_expert_bytes", "dma_compute_overlap_cycles",
    ):
        print("  {}: {}".format(field, getattr(summary, field)))
    for name, path in {**paths, **plot_paths}.items():
        print("  {}: {}".format(name, path))
    return 1 if summary.kv_capacity_exceeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
