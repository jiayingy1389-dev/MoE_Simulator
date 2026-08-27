"""Timeline table reporting for existing MoE V0 JSON results."""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ReportError(ValueError):
    """Raised when timeline report input is missing or inconsistent."""


@dataclass(frozen=True)
class TimelineRow:
    event_id: int
    start_cycle: int
    end_cycle: int
    duration: int
    token_id: Optional[int]
    stage: str
    resource: str
    expert_id: Optional[int]
    tile_id: Optional[int]
    fixed_reserved_bytes: int
    workspace_reserved_bytes: int
    on_chip_kv_bytes: int
    on_chip_total_reserved_bytes: int
    on_chip_capacity_bytes: int
    on_chip_utilization_percent: float
    bytes_transferred: int
    effective_bandwidth_bytes_per_cycle: float
    peak_bandwidth_bytes_per_cycle: float
    bandwidth_utilization_percent: float

    def to_csv_dict(self) -> Dict[str, Any]:
        return asdict(self)


COLUMNS = (
    "event_id",
    "start_cycle",
    "end_cycle",
    "duration",
    "token_id",
    "stage",
    "resource",
    "expert_id",
    "tile_id",
    "fixed_reserved_bytes",
    "workspace_reserved_bytes",
    "on_chip_kv_bytes",
    "on_chip_total_reserved_bytes",
    "on_chip_capacity_bytes",
    "on_chip_utilization_percent",
    "bytes_transferred",
    "effective_bandwidth_bytes_per_cycle",
    "peak_bandwidth_bytes_per_cycle",
    "bandwidth_utilization_percent",
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError("{} must be a mapping".format(path))
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportError("{} must be a sequence".format(path))
    return value


def _required(mapping: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in mapping:
        raise ReportError("{}.{} is required".format(path, field))
    return mapping[field]


def _number(mapping: Mapping[str, Any], field: str, path: str) -> float:
    value = _required(mapping, field, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError("{}.{} must be numeric".format(path, field))
    return value


def _integer(mapping: Mapping[str, Any], field: str, path: str) -> int:
    value = _required(mapping, field, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError("{}.{} must be an integer".format(path, field))
    return value


def _optional_integer(mapping: Mapping[str, Any], field: str, path: str) -> Optional[int]:
    value = _required(mapping, field, path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError("{}.{} must be an integer or null".format(path, field))
    return value


def build_rows(payload: Mapping[str, Any]) -> List[TimelineRow]:
    root = _mapping(payload, "root")
    config = _mapping(_required(root, "config", "root"), "config")
    hardware = _mapping(
        _required(config, "hardware", "config"), "config.hardware"
    )
    timeline = _sequence(_required(root, "timeline", "root"), "timeline")

    fixed = _integer(hardware, "fixed_reserved_bytes", "config.hardware")
    workspace = _integer(hardware, "workspace_bytes", "config.hardware")
    capacity = _integer(hardware, "on_chip_capacity_bytes", "config.hardware")
    peak_bandwidth = _number(
        hardware, "off_chip_bytes_per_cycle", "config.hardware"
    )
    if fixed < 0:
        raise ReportError("config.hardware.fixed_reserved_bytes must be non-negative")
    if workspace < 0:
        raise ReportError("config.hardware.workspace_bytes must be non-negative")
    if capacity <= 0:
        raise ReportError("config.hardware.on_chip_capacity_bytes must be positive")
    if peak_bandwidth <= 0:
        raise ReportError("config.hardware.off_chip_bytes_per_cycle must be positive")

    rows: List[TimelineRow] = []
    for index, raw_event in enumerate(timeline):
        path = "timeline[{}]".format(index)
        event = _mapping(raw_event, path)
        event_id = _integer(event, "event_id", path)
        start_cycle = _integer(event, "start_cycle", path)
        end_cycle = _integer(event, "end_cycle", path)
        duration = _integer(event, "duration", path)
        token_id = _optional_integer(event, "token_id", path)
        expert_id = _optional_integer(event, "expert_id", path)
        tile_id = _optional_integer(event, "tile_id", path)
        stage = _required(event, "stage", path)
        resource = _required(event, "resource", path)
        if not isinstance(stage, str):
            raise ReportError("{}.stage must be a string".format(path))
        if resource not in ("memory", "compute", "state"):
            raise ReportError("{}.resource must be memory, compute, or state".format(path))
        bytes_transferred = _integer(event, "bytes_transferred", path)
        on_chip_kv = _integer(event, "on_chip_kv_bytes", path)
        if duration < 0:
            raise ReportError("{}.duration must be non-negative".format(path))
        if end_cycle - start_cycle != duration:
            raise ReportError("{}.duration does not match its cycle interval".format(path))
        if bytes_transferred < 0:
            raise ReportError("{}.bytes_transferred must be non-negative".format(path))
        if on_chip_kv < 0:
            raise ReportError("{}.on_chip_kv_bytes must be non-negative".format(path))
        if resource == "memory" and duration <= 0:
            raise ReportError("{}.duration must be positive for Memory events".format(path))

        total = fixed + workspace + on_chip_kv
        if total > capacity:
            raise ReportError(
                "{} on-chip total {} exceeds capacity {}".format(path, total, capacity)
            )
        effective = bytes_transferred / duration if resource == "memory" else 0.0
        rows.append(
            TimelineRow(
                event_id=event_id,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                duration=duration,
                token_id=token_id,
                stage=stage,
                resource=resource,
                expert_id=expert_id,
                tile_id=tile_id,
                fixed_reserved_bytes=fixed,
                workspace_reserved_bytes=workspace,
                on_chip_kv_bytes=on_chip_kv,
                on_chip_total_reserved_bytes=total,
                on_chip_capacity_bytes=capacity,
                on_chip_utilization_percent=100.0 * total / capacity,
                bytes_transferred=bytes_transferred,
                effective_bandwidth_bytes_per_cycle=effective,
                peak_bandwidth_bytes_per_cycle=float(peak_bandwidth),
                bandwidth_utilization_percent=100.0 * effective / peak_bandwidth,
            )
        )
    return rows


def load_report(path: Path) -> List[TimelineRow]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError("cannot load report input {}: {}".format(path, exc)) from exc
    return build_rows(payload)


def write_csv(rows: Sequence[TimelineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _markdown_value(column: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        rendered = "{:.2f}".format(value)
        if column.endswith("_percent"):
            return rendered + "%"
        return rendered
    return str(value)


def write_markdown(rows: Sequence[TimelineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MoE V0 Timeline Table",
        "",
        "> On-chip total is fixed reserved + workspace reserved + on-chip KV.  ",
        "> Workspace is reserved capacity, not measured dynamic Expert-buffer occupancy.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        values = row.to_csv_dict()
        lines.append(
            "| "
            + " | ".join(_markdown_value(column, values[column]) for column in COLUMNS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MoE V0 JSON events as CSV and Markdown timeline tables"
    )
    parser.add_argument("input", type=Path, help="existing simulator JSON result")
    parser.add_argument("--csv", required=True, type=Path, help="CSV output path")
    parser.add_argument(
        "--markdown", required=True, type=Path, help="Markdown output path"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_report(args.input)
    except ReportError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print("wrote {} timeline rows".format(len(rows)))
    print("  csv: {}".format(args.csv))
    print("  markdown: {}".format(args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
