import csv
import json
from pathlib import Path

from simulator.v1.config import V1Config
from simulator.v1.outputs import RESOURCE_COLUMNS, TOKEN_COLUMNS, write_v1_outputs
from simulator.v1.plots import write_v1_plots
from simulator.v1.simulator import V1Simulator

from .test_simulator import small_config


def test_writers_create_deterministic_csv_json_and_valid_pngs(
    tmp_path, v1_config_dict
):
    config = small_config(v1_config_dict)
    result = V1Simulator(config).run()
    write_v1_outputs(result, tmp_path)
    write_v1_plots(result, config, tmp_path)

    token_path = tmp_path / "v1_token_state.csv"
    resource_path = tmp_path / "v1_resource_timeline.csv"
    summary_path = tmp_path / "v1_summary.json"
    with token_path.open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(TOKEN_COLUMNS)
    with resource_path.open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(RESOURCE_COLUMNS)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "OK"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {
        "peak_expert_cache_bytes", "expert_cache_hits",
        "expert_cache_misses", "expert_cache_hit_rate", "expert_evictions",
        "kv_triggered_expert_evictions",
    } <= set(summary)

    first_contents = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    write_v1_outputs(result, tmp_path)
    write_v1_plots(result, config, tmp_path)
    assert first_contents == {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    for name in (
        "v1_kv_and_expert_cache_over_time.png",
        "v1_bandwidth_over_time.png",
    ):
        payload = (tmp_path / name).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1000


def test_default_config_file_contains_confirmed_model_values():
    path = Path(__file__).parents[2] / "configs" / "v1_qwen_synthetic.yaml"
    config = V1Config.from_dict(__import__("yaml").safe_load(path.read_text()))
    assert config.model.num_layers == 24
    assert config.model.num_routed_experts == 60
    assert config.model.top_k == 4
    assert config.request.prompt_tokens == 32
    assert config.request.decode_tokens == 96
    assert config.hardware.on_chip_memory_mib == 32
    assert config.hardware.off_chip_bandwidth_gbps == 200
    assert config.prefetch.prediction_accuracy == 0.8
