import pytest

from simulator.v1.config import V1Config, V1ConfigError


def test_default_units_and_rates(v1_config):
    assert v1_config.hardware.on_chip_memory_bytes == 32 * 1024 * 1024
    assert v1_config.hardware.fixed_reserved_bytes == 3 * 1024 * 1024
    assert v1_config.hardware.expert_workspace_bytes == 5 * 1024 * 1024
    assert v1_config.hardware.off_chip_bytes_per_cycle == pytest.approx(200e9 / 300e6)


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("model", "batch_size"), 2, "batch_size"),
        (("model", "num_requests"), 2, "num_requests"),
        (("prefetch", "prefetch_distance"), 2, "prefetch_distance"),
        (("prefetch", "prediction_accuracy"), 1.1, "prediction_accuracy"),
        (("hardware", "compute_ops_per_cycle"), 0, "compute_ops_per_cycle"),
        (("hardware", "off_chip_bandwidth_gbps"), 0, "off_chip_bandwidth_gbps"),
        (("hardware", "expert_workspace_mib"), 4, "expert_workspace"),
    ],
)
def test_rejects_invalid_configuration(v1_config_dict, path, value, match):
    v1_config_dict[path[0]][path[1]] = value
    with pytest.raises(V1ConfigError, match=match):
        V1Config.from_dict(v1_config_dict)


def test_rejects_prompt_kv_that_cannot_fit(v1_config_dict):
    v1_config_dict["request"]["prompt_tokens"] = 200
    with pytest.raises(V1ConfigError, match="prompt.*KV"):
        V1Config.from_dict(v1_config_dict)
