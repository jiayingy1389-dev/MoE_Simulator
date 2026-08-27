import copy
import json
from pathlib import Path

import pytest

from simulator.report import (
    ReportError,
    build_rows,
    main,
    write_csv,
    write_markdown,
)


@pytest.fixture
def timeline_payload():
    path = Path(__file__).resolve().parents[1] / "outputs" / "v0_timeline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_builds_one_ordered_row_per_event(timeline_payload):
    rows = build_rows(timeline_payload)
    assert len(rows) == len(timeline_payload["timeline"]) == 85
    assert [row.event_id for row in rows] == list(range(85))


def test_calculates_memory_reservation_and_utilization(timeline_payload):
    rows = build_rows(timeline_payload)
    first_dma = next(row for row in rows if row.stage == "kv_write")
    assert first_dma.fixed_reserved_bytes == 128
    assert first_dma.workspace_reserved_bytes == 128
    assert first_dma.on_chip_kv_bytes == 32
    assert first_dma.on_chip_total_reserved_bytes == 288
    assert first_dma.on_chip_capacity_bytes == 288
    assert first_dma.on_chip_utilization_percent == pytest.approx(100.0)

    final = rows[-1]
    assert final.stage == "release_request_kv"
    assert final.on_chip_kv_bytes == 0
    assert final.on_chip_total_reserved_bytes == 256


def test_calculates_event_average_bandwidth(timeline_payload):
    rows = build_rows(timeline_payload)
    first_dma = next(row for row in rows if row.stage == "kv_write")
    assert first_dma.bytes_transferred == 16
    assert first_dma.duration == 3
    assert first_dma.effective_bandwidth_bytes_per_cycle == pytest.approx(16 / 3)
    assert first_dma.peak_bandwidth_bytes_per_cycle == 16
    assert first_dma.bandwidth_utilization_percent == pytest.approx(100 / 3)

    assert rows[0].resource == "compute"
    assert rows[0].effective_bandwidth_bytes_per_cycle == 0
    assert rows[1].resource == "state"
    assert rows[1].bandwidth_utilization_percent == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.pop("config"), "config"),
        (lambda payload: payload["config"].pop("hardware"), "config.hardware"),
        (lambda payload: payload.pop("timeline"), "timeline"),
        (lambda payload: payload["timeline"][0].pop("stage"), "timeline.*stage"),
        (
            lambda payload: payload["config"]["hardware"].update(
                {"on_chip_capacity_bytes": 0}
            ),
            "on_chip_capacity_bytes",
        ),
        (
            lambda payload: payload["config"]["hardware"].update(
                {"off_chip_bytes_per_cycle": 0}
            ),
            "off_chip_bytes_per_cycle",
        ),
        (
            lambda payload: payload["timeline"][2].update({"duration": 0}),
            "duration",
        ),
    ],
)
def test_rejects_invalid_report_input(timeline_payload, mutation, match):
    invalid = copy.deepcopy(timeline_payload)
    mutation(invalid)
    with pytest.raises(ReportError, match=match):
        build_rows(invalid)


def test_writes_deterministic_csv_and_markdown(timeline_payload, tmp_path):
    rows = build_rows(timeline_payload)
    csv_first = tmp_path / "first.csv"
    csv_second = tmp_path / "second.csv"
    md_first = tmp_path / "first.md"
    md_second = tmp_path / "second.md"
    write_csv(rows, csv_first)
    write_csv(rows, csv_second)
    write_markdown(rows, md_first)
    write_markdown(rows, md_second)

    assert csv_first.read_bytes() == csv_second.read_bytes()
    assert md_first.read_bytes() == md_second.read_bytes()
    csv_lines = csv_first.read_text(encoding="utf-8").splitlines()
    markdown = md_first.read_text(encoding="utf-8")
    markdown_table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(csv_lines) == 86
    assert len(markdown_table_lines) == 87
    assert "workspace_reserved_bytes" in csv_lines[0]
    assert "Workspace is reserved capacity" in markdown
    kv_write_line = next(line for line in markdown_table_lines if "kv_write" in line)
    assert "5.33" in kv_write_line
    assert "33.33%" in kv_write_line


def test_report_cli_writes_both_files(timeline_payload, tmp_path, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(timeline_payload), encoding="utf-8")
    csv_path = tmp_path / "table.csv"
    markdown_path = tmp_path / "table.md"
    assert main(
        [
            str(input_path),
            "--csv",
            str(csv_path),
            "--markdown",
            str(markdown_path),
        ]
    ) == 0
    assert csv_path.exists()
    assert markdown_path.exists()
    assert "85 timeline rows" in capsys.readouterr().out


def test_report_cli_handles_invalid_input_without_traceback(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert main(
        [
            str(invalid),
            "--csv",
            str(tmp_path / "unused.csv"),
            "--markdown",
            str(tmp_path / "unused.md"),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
