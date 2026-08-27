import json
from pathlib import Path

from simulator.cli import main


def test_synthetic_cli_writes_deterministic_json_and_summary(tmp_path, capsys):
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "v0_synthetic.yaml"
    output_path = tmp_path / "timeline.json"
    assert main([str(config_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["timeline"]
    assert all("shape" in event for event in payload["timeline"])
    assert payload["timeline"][-1]["stage"] == "release_request_kv"
    assert payload["summary"]["first_kv_spill_token"] == 0
    assert payload["summary"]["memory_cycles"] + payload["summary"]["compute_cycles"] == payload["summary"]["total_cycles"]
    stdout = capsys.readouterr().out
    assert "total_cycles" in stdout
    assert "token 0 latency" in stdout


def test_cli_reports_configuration_errors_without_traceback(tmp_path, capsys):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("model: {}\n", encoding="utf-8")
    assert main([str(invalid), "--output", str(tmp_path / "unused.json")]) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
