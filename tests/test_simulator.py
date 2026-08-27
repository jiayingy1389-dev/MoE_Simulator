from simulator.config import SimulationConfig
from simulator.simulator import MoESimulator


def spill_config(valid_config_dict):
    valid_config_dict["hardware"]["on_chip_capacity_bytes"] = 288
    return SimulationConfig.from_dict(valid_config_dict)


def token_events(result, token_id):
    return [event for event in result.timeline if event.token_id == token_id]


def test_token_and_expert_execution_order(valid_config_dict):
    result = MoESimulator(spill_config(valid_config_dict)).run()
    events = token_events(result, 0)
    stages = [event.stage for event in events]
    required_prefix = [
        "qkv_attention_prepare",
        "place_token_kv",
        "kv_write",
        "attention_kv_read",
        "attention_compute",
        "router_compute",
    ]
    assert stages[:6] == required_prefix
    assert stages[-2:] == ["topk_merge", "token_complete"]

    first_expert = [event for event in events if event.expert_id == 0]
    assert first_expert[0].stage == "expert_partial_sum_allocate"
    assert first_expert[-1].stage == "expert_release"
    for tile_id in range(3):
        assert [
            event.stage
            for event in first_expert
            if event.tile_id == tile_id
        ] == [
            "expert_gu_weight_read",
            "expert_gu_compute",
            "expert_nonlinear",
            "expert_down_weight_read",
            "expert_down_compute",
        ]


def test_router_attention_and_expert_operations_follow_formulas(valid_config_dict):
    config = spill_config(valid_config_dict)
    result = MoESimulator(config).run()
    first = token_events(result, 0)
    router = next(event for event in first if event.stage == "router_compute")
    attention = next(event for event in first if event.stage == "attention_compute")
    assert router.operations == 2 * config.model.H * config.model.E
    assert router.shape == {"H": config.model.H, "E": config.model.E}
    assert attention.operations == (
        config.model.attention_base_ops
        + (config.request.initial_prompt_length + 1)
        * config.model.attention_ops_per_context_token
    )
    assert attention.shape == {
        "context_length": config.request.initial_prompt_length + 1,
        "H": config.model.H,
    }
    expert_zero_ops = sum(event.operations for event in first if event.expert_id == 0)
    assert expert_zero_ops == 6 * config.model.H * config.model.F
    tile = next(event for event in first if event.stage == "expert_gu_compute")
    assert tile.shape == {"H": config.model.H, "F_i": config.model.f_tile_size}


def test_repeated_expert_weights_are_reloaded_for_each_token(valid_config):
    result = MoESimulator(valid_config).run()
    expected_per_selection = 3 * valid_config.model.H * valid_config.model.F
    for token_id in (0, 1):
        loaded = sum(
            event.bytes_transferred
            for event in token_events(result, token_id)
            if event.expert_id == 1 and "weight_read" in event.stage
        )
        assert loaded == expected_per_selection


def test_kv_reads_grow_after_spill(valid_config_dict):
    result = MoESimulator(spill_config(valid_config_dict)).run()
    reads = [event.bytes_transferred for event in result.timeline if event.stage == "attention_kv_read"]
    assert reads == [16, 32]
    assert result.summary.off_chip_kv_write_bytes == 32
    assert result.summary.first_kv_spill_token == 0


def test_ocm_capacity_is_never_exceeded(valid_config_dict):
    config = spill_config(valid_config_dict)
    result = MoESimulator(config).run()
    reserved = config.hardware.fixed_reserved_bytes + config.hardware.workspace_bytes
    assert all(
        event.on_chip_kv_bytes + reserved <= config.hardware.on_chip_capacity_bytes
        for event in result.timeline
    )


def test_request_release_zeros_current_kv_and_retains_peaks(valid_config_dict):
    result = MoESimulator(spill_config(valid_config_dict)).run()
    final = result.timeline[-1]
    assert final.stage == "release_request_kv"
    assert final.duration == 0
    assert final.on_chip_kv_bytes == 0
    assert final.off_chip_kv_bytes == 0
    assert result.summary.on_chip_kv_peak_bytes == 32
    assert result.summary.off_chip_kv_peak_bytes == 32


def test_summary_is_derived_from_timeline(valid_config_dict):
    result = MoESimulator(spill_config(valid_config_dict)).run()
    memory_cycles = sum(event.duration for event in result.timeline if event.resource == "memory")
    compute_cycles = sum(event.duration for event in result.timeline if event.resource == "compute")
    assert result.summary.memory_cycles == memory_cycles
    assert result.summary.compute_cycles == compute_cycles
    assert result.summary.total_cycles == memory_cycles + compute_cycles
    assert result.summary.nonlinear_cycles == sum(
        event.duration for event in result.timeline if event.stage == "expert_nonlinear"
    )
    assert result.summary.off_chip_kv_read_bytes == sum(
        event.bytes_transferred for event in result.timeline if event.stage == "attention_kv_read"
    )
    assert result.summary.expert_weight_read_bytes == sum(
        event.bytes_transferred for event in result.timeline if "weight_read" in event.stage
    )


def test_token_latency_uses_first_event_through_completion(valid_config):
    result = MoESimulator(valid_config).run()
    for summary in result.tokens:
        events = token_events(result, summary.token_id)
        completion = next(event for event in events if event.stage == "token_complete")
        assert summary.start_cycle == events[0].start_cycle
        assert summary.end_cycle == completion.end_cycle
        assert summary.latency == summary.end_cycle - summary.start_cycle


def test_identical_inputs_are_fully_deterministic(valid_config):
    first = MoESimulator(valid_config).run().to_dict()
    second = MoESimulator(valid_config).run().to_dict()
    assert first == second
