# MoE Timeline Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic CSV and Markdown timeline tables from the existing V0 JSON result, showing stage, confirmed on-chip memory accounting, and event-average bandwidth.

**Architecture:** A focused `simulator.report` module loads and validates the existing JSON, converts every timeline event into one flat report row, and renders the same rows to CSV and Markdown. Formulas remain presentation-only and do not modify simulator results.

**Tech Stack:** Python 3.9 standard library (`argparse`, `csv`, `json`, `dataclasses`, `pathlib`) and pytest 7.

## Global Constraints

- Preserve all source events and their order; do not merge intervals.
- On-chip total is fixed reservation plus workspace reservation plus current on-chip KV.
- Workspace is labeled as reserved, not measured dynamic Expert-buffer occupancy.
- Effective bandwidth is bytes divided by event duration for Memory events only.
- Compute and state events have zero effective bandwidth.
- CSV numeric fields contain numbers without unit or percent suffixes.
- Markdown numeric rates and percentages use two decimal places.
- Do not change V0 simulation timing or behavior.

---

### Task 1: Timeline Report Rows and Formulas

**Files:**
- Create: `simulator/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: `ReportError`, `TimelineRow`, `load_report(path: Path) -> List[TimelineRow]`.
- Produces: `build_rows(payload: Mapping[str, Any]) -> List[TimelineRow]`.
- `TimelineRow.to_csv_dict()` returns flat JSON/CSV-native values.

- [ ] **Step 1: Write failing row tests**

Load `outputs/v0_timeline.json`, call `build_rows()`, and assert:

```python
assert len(rows) == len(payload["timeline"]) == 85
assert [row.event_id for row in rows] == list(range(85))

first_dma = next(row for row in rows if row.stage == "kv_write")
assert first_dma.bytes_transferred == 16
assert first_dma.duration == 3
assert first_dma.effective_bandwidth_bytes_per_cycle == pytest.approx(16 / 3)
assert first_dma.bandwidth_utilization_percent == pytest.approx(100 / 3)

compute = rows[0]
assert compute.resource == "compute"
assert compute.effective_bandwidth_bytes_per_cycle == 0

assert first_dma.fixed_reserved_bytes == 128
assert first_dma.workspace_reserved_bytes == 128
assert first_dma.on_chip_kv_bytes == 32
assert first_dma.on_chip_total_reserved_bytes == 288
assert first_dma.on_chip_utilization_percent == 100

final = rows[-1]
assert final.stage == "release_request_kv"
assert final.on_chip_kv_bytes == 0
assert final.on_chip_total_reserved_bytes == 256
```

Add parameterized invalid-payload tests for missing `config.hardware`, missing `timeline`, missing event fields, non-positive capacity, and positive-duration Memory events with zero duration. Each must raise `ReportError` naming the missing or invalid field.

- [ ] **Step 2: Run tests and verify RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_report.py -q`

Expected: collection fails because `simulator.report` does not exist.

- [ ] **Step 3: Implement validated row conversion**

Define a frozen dataclass with the 19 columns from the design. `build_rows()` extracts:

```python
fixed = hardware["fixed_reserved_bytes"]
workspace = hardware["workspace_bytes"]
capacity = hardware["on_chip_capacity_bytes"]
peak_bandwidth = hardware["off_chip_bytes_per_cycle"]
total = fixed + workspace + event["on_chip_kv_bytes"]
effective = (
    event["bytes_transferred"] / event["duration"]
    if event["resource"] == "memory" else 0.0
)
bandwidth_utilization = 100.0 * effective / peak_bandwidth
on_chip_utilization = 100.0 * total / capacity
```

Reject a Memory event with non-positive duration, required numeric fields with the wrong type, non-positive capacity or peak bandwidth, and totals exceeding capacity. Preserve nullable IDs as `None`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_report.py -q`

Expected: all row and validation tests pass.

- [ ] **Step 5: Commit Task 1**

```text
git add simulator/report.py tests/test_report.py
git commit -m "feat: calculate MoE timeline report rows"
```

---

### Task 2: CSV/Markdown Writers and Generated Artifacts

**Files:**
- Modify: `simulator/report.py`
- Modify: `tests/test_report.py`
- Modify: `README.md`
- Create: `outputs/v0_timeline_table.csv`
- Create: `outputs/v0_timeline_table.md`

**Interfaces:**
- Produces: `write_csv(rows, path)`, `write_markdown(rows, path)`, and `main(argv=None) -> int`.
- CLI: `python -m simulator.report INPUT --csv CSV_PATH --markdown MARKDOWN_PATH`.

- [ ] **Step 1: Write failing renderer and CLI tests**

Use `tmp_path` to render both formats twice. Assert byte-for-byte deterministic output, 86 CSV lines including header, 87 Markdown table lines including header/separator, required column names, a visible workspace-reservation note, `5.33` effective bandwidth for the first KV write, and `33.33%` utilization in Markdown. Assert CLI returns 2 and prints `error:` without traceback for invalid input.

- [ ] **Step 2: Run tests and verify RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_report.py -q`

Expected: renderer imports or assertions fail because writers and CLI are not implemented.

- [ ] **Step 3: Implement deterministic writers and CLI**

Use `csv.DictWriter` with an explicit column tuple and `lineterminator="\n"`. Markdown uses the same order, renders `None` as `-`, and formats floats to two decimals. Prepend this note before the table:

```text
> On-chip total is fixed reserved + workspace reserved + on-chip KV.
> Workspace is reserved capacity, not measured dynamic Expert-buffer occupancy.
```

Because the note adds lines, renderer tests count table rows separately by selecting lines starting with `|`.

The CLI loads once, writes both requested paths, prints row count and both paths, catches only `ReportError`, and returns 2 on report-data errors.

- [ ] **Step 4: Run the complete test suite**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q`

Expected: all existing and new tests pass without warnings.

- [ ] **Step 5: Generate and inspect artifacts**

Run:

```powershell
python -m simulator.report outputs/v0_timeline.json `
  --csv outputs/v0_timeline_table.csv `
  --markdown outputs/v0_timeline_table.md
```

Expected: 85 rows reported, both files created, first DMA bandwidth is `5.33 B/cycle`, its utilization is `33.33%`, and the final row has 256 bytes total reserved after KV release.

- [ ] **Step 6: Document the report command and accounting note**

Add the command, output links, three displayed concepts, formulas, and workspace-reservation limitation to README. Do not describe workspace as dynamic occupancy.

- [ ] **Step 7: Final verification and commit**

Run: `git diff --check`

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q`

Expected: no whitespace errors and all tests pass.

```text
git add simulator/report.py tests/test_report.py README.md outputs/v0_timeline_table.csv outputs/v0_timeline_table.md
git commit -m "feat: export MoE timeline tables"
```
