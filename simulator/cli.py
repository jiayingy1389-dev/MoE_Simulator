"""Command-line interface for the MoE V0 simulator."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, load_config
from .simulator import MoESimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the single-layer MoE V0 simulator")
    parser.add_argument("config", type=Path, help="YAML configuration path")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        result = MoESimulator(config).run()
        payload = result.to_dict()
        payload["config"] = asdict(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ConfigError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    summary = result.summary
    print("MoE V0 simulation summary")
    print("  total_cycles: {}".format(summary.total_cycles))
    print("  memory_cycles: {}".format(summary.memory_cycles))
    print("  compute_cycles: {}".format(summary.compute_cycles))
    print("  nonlinear_cycles: {}".format(summary.nonlinear_cycles))
    print("  off_chip_kv_read_bytes: {}".format(summary.off_chip_kv_read_bytes))
    print("  off_chip_kv_write_bytes: {}".format(summary.off_chip_kv_write_bytes))
    print("  expert_weight_read_bytes: {}".format(summary.expert_weight_read_bytes))
    print("  on_chip_kv_peak_bytes: {}".format(summary.on_chip_kv_peak_bytes))
    print("  off_chip_kv_peak_bytes: {}".format(summary.off_chip_kv_peak_bytes))
    print("  first_kv_spill_token: {}".format(summary.first_kv_spill_token))
    for token in result.tokens:
        print("  token {} latency: {} cycles".format(token.token_id, token.latency))
    print("  timeline: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
