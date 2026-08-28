"""Deterministic CSV and JSON writers for V1 results."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence

from .model import V1Result


TOKEN_COLUMNS = (
    "token_id", "context_length", "start_cycle", "finish_cycle",
    "kv_occupancy_bytes", "expert_cache_capacity_bytes",
    "expert_cache_occupancy_bytes", "resident_expert_count",
    "kv_eviction_count", "total_eviction_count",
)
RESOURCE_COLUMNS = (
    "start_cycle", "end_cycle", "resource", "operation", "token_id",
    "layer_id", "expert_id", "bytes", "prefetch_or_demand",
    "prediction_correct",
)


def _write_csv(path: Path, columns: Sequence[str], rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_v1_outputs(result: V1Result, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "v1_token_state.csv"
    resource_path = output_dir / "v1_resource_timeline.csv"
    summary_path = output_dir / "v1_summary.json"
    _write_csv(token_path, TOKEN_COLUMNS, [
        {field: getattr(state, field) for field in TOKEN_COLUMNS}
        for state in result.token_states
    ])
    events = sorted(result.events,
                    key=lambda event: (event.start_cycle, event.resource,
                                       event.sequence_id))
    _write_csv(resource_path, RESOURCE_COLUMNS, [
        {field: getattr(event, field) for field in RESOURCE_COLUMNS}
        for event in events
    ])
    summary_path.write_text(
        json.dumps(asdict(result.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"tokens": token_path, "resources": resource_path,
            "summary": summary_path}
