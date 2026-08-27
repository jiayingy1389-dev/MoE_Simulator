import pytest

from simulator.config import ConfigError, SimulationConfig


def test_accepts_valid_configuration(valid_config_dict):
    config = SimulationConfig.from_dict(valid_config_dict)
    assert config.model.H == 8
    assert config.request.routing_trace == ((0, 1), (1, 2))


def test_rejects_capacity_partition(valid_config_dict):
    valid_config_dict["hardware"]["fixed_reserved_bytes"] = 400
    with pytest.raises(ConfigError, match="fixed_reserved_bytes.*workspace_bytes"):
        SimulationConfig.from_dict(valid_config_dict)


@pytest.mark.parametrize(
    "section,field",
    [
        ("hardware", "off_chip_bytes_per_cycle"),
        ("hardware", "compute_ops_per_cycle"),
        ("model", "H"),
        ("model", "weight_bits"),
    ],
)
def test_rejects_nonpositive_values(valid_config_dict, section, field):
    valid_config_dict[section][field] = 0
    with pytest.raises(ConfigError, match=field):
        SimulationConfig.from_dict(valid_config_dict)


def test_rejects_negative_cycle_cost(valid_config_dict):
    valid_config_dict["hardware"]["dma_startup_cycles"] = -1
    with pytest.raises(ConfigError, match="dma_startup_cycles"):
        SimulationConfig.from_dict(valid_config_dict)


@pytest.mark.parametrize(
    "trace",
    [
        [[0, 0], [1, 2]],
        [[0, 1]],
        [[0, 1], [1, 7]],
        [[0], [1, 2]],
    ],
)
def test_rejects_bad_routing_trace(valid_config_dict, trace):
    valid_config_dict["request"]["routing_trace"] = trace
    with pytest.raises(ConfigError, match="routing_trace"):
        SimulationConfig.from_dict(valid_config_dict)


def test_rejects_topk_larger_than_expert_count(valid_config_dict):
    valid_config_dict["model"]["K"] = 5
    with pytest.raises(ConfigError, match="K.*E"):
        SimulationConfig.from_dict(valid_config_dict)


def test_rejects_missing_field_with_clear_path(valid_config_dict):
    del valid_config_dict["model"]["H"]
    with pytest.raises(ConfigError, match="model.H"):
        SimulationConfig.from_dict(valid_config_dict)
