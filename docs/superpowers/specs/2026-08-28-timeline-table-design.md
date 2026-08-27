# MoE V0 Timeline Table Design

## Goal

Transform the existing synthetic result in `outputs/v0_timeline.json` into two human-readable timeline tables that show, for every event interval:

1. the current execution stage;
2. current on-chip memory occupation/reservation;
3. current effective off-chip bandwidth and utilization.

The table is a presentation of the existing V0 result. It does not change simulation order, timing, formulas, or resource behavior.

## Outputs

- `outputs/v0_timeline_table.csv`: the complete table for Excel, filtering, and plotting.
- `outputs/v0_timeline_table.md`: the same rows as a directly readable Markdown table.

Both files preserve all 85 timeline events in event order. No adjacent stages are merged, so Expert and F-tile detail remains visible.

## Columns

Each row contains:

```text
event_id
start_cycle
end_cycle
duration
token_id
stage
resource
expert_id
tile_id
fixed_reserved_bytes
workspace_reserved_bytes
on_chip_kv_bytes
on_chip_total_reserved_bytes
on_chip_capacity_bytes
on_chip_utilization_percent
bytes_transferred
effective_bandwidth_bytes_per_cycle
peak_bandwidth_bytes_per_cycle
bandwidth_utilization_percent
```

Nullable token, expert, and tile values are rendered as `-` in Markdown and as empty cells in CSV.

## Memory Accounting

The table reports:

```text
on_chip_total_reserved_bytes
    = fixed_reserved_bytes
    + workspace_reserved_bytes
    + on_chip_kv_bytes
```

`fixed_reserved_bytes` and `workspace_reserved_bytes` come from the synthetic configuration and remain reserved for the full request. `on_chip_kv_bytes` comes from each event snapshot and changes as KV is placed and released.

V0 does not track the dynamically live byte count of every Expert temporary buffer. Therefore `workspace_reserved_bytes` is a reservation, not measured instantaneous buffer occupancy. The column names and Markdown note must make this distinction explicit.

On-chip utilization is:

```text
100 * on_chip_total_reserved_bytes / on_chip_capacity_bytes
```

## Bandwidth Accounting

For positive-duration Memory events:

```text
effective_bandwidth_bytes_per_cycle
    = bytes_transferred / duration

bandwidth_utilization_percent
    = 100
    * effective_bandwidth_bytes_per_cycle
    / off_chip_bytes_per_cycle
```

This is the event-average effective bandwidth and includes DMA startup cycles in the denominator. Compute and state events report zero effective bandwidth and zero utilization. The configured `off_chip_bytes_per_cycle` is shown separately as peak bandwidth.

Values are formatted to two decimal places in Markdown. CSV stores numeric values without percent signs so spreadsheet tools can process them.

## Implementation

Add a small report module and CLI:

```text
simulator/report.py
tests/test_report.py
```

The command is:

```powershell
python -m simulator.report outputs/v0_timeline.json \
  --csv outputs/v0_timeline_table.csv \
  --markdown outputs/v0_timeline_table.md
```

The report loader validates required configuration, timeline, and event fields before writing either output. Generation is deterministic for identical JSON input.

## Tests

Tests verify:

- one output row per source event, in the same order;
- memory totals and utilization use the confirmed reservation formula;
- Memory-event bandwidth uses bytes divided by duration;
- DMA startup time lowers effective bandwidth below configured peak where applicable;
- Compute and state bandwidth is zero;
- request-end KV release reduces the final total to fixed plus workspace reservations;
- CSV and Markdown files are produced deterministically;
- missing required input fields produce clear errors.
