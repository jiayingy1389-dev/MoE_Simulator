import pytest

from simulator.compute import (
    compute_cycles,
    dma_cycles,
    expert_tile_metrics,
    f_tiles,
    required_workspace_bytes,
)


def test_tail_tile_conserves_f():
    assert f_tiles(10, 4) == [4, 4, 2]
    assert sum(f_tiles(10, 4)) == 10


def test_expert_totals_match_theory(valid_config):
    model = valid_config.model
    metrics = [expert_tile_metrics(model, size) for size in f_tiles(model.F, model.f_tile_size)]
    assert sum(item.gu_weight_bytes for item in metrics) == 2 * model.H * model.F
    assert sum(item.down_weight_bytes for item in metrics) == model.H * model.F
    assert sum(item.gu_operations for item in metrics) == 4 * model.H * model.F
    assert sum(item.down_operations for item in metrics) == 2 * model.H * model.F


def test_timing_formulas_round_up(valid_config):
    hardware = valid_config.hardware
    assert dma_cycles(17, hardware) == hardware.dma_startup_cycles + 2
    assert compute_cycles(33, hardware) == hardware.compute_startup_cycles + 2


def test_workspace_formula_counts_simultaneously_live_buffers(valid_config):
    assert required_workspace_bytes(valid_config.model) == 112


def test_configuration_rejects_insufficient_workspace(valid_config_dict):
    from simulator.config import ConfigError, SimulationConfig

    valid_config_dict["hardware"]["workspace_bytes"] = 111
    with pytest.raises(ConfigError, match="workspace_bytes.*112"):
        SimulationConfig.from_dict(valid_config_dict)
