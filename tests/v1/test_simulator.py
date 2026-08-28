from simulator.v1.config import V1Config
from simulator.v1.model import COMPUTE, OFF_CHIP_DMA, ON_CHIP_KV_READ
from simulator.v1.simulator import V1Simulator


def small_config(v1_config_dict, prefetch=True, accuracy=1.0):
    model = v1_config_dict["model"]
    model.update(
        {
            "num_layers": 3,
            "hidden_size": 8,
            "expert_intermediate_size": 4,
            "num_routed_experts": 6,
            "top_k": 2,
            "num_kv_heads": 1,
            "head_dim": 4,
            "weight_bits": 8,
            "kv_bits": 8,
            "attention_base_ops": 0,
            "attention_ops_per_context_token": 4,
        }
    )
    v1_config_dict["request"].update({"prompt_tokens": 2, "decode_tokens": 3})
    v1_config_dict["hardware"].update(
        {
            "on_chip_memory_mib": 8,
            "fixed_reserved_mib": 1,
            "expert_workspace_mib": 1,
            "clock_frequency_mhz": 100,
            "off_chip_bandwidth_gbps": 1,
            "on_chip_read_bytes_per_cycle": 8,
            "compute_ops_per_cycle": 16,
        }
    )
    v1_config_dict["prefetch"].update(
        {"prefetch_enabled": prefetch, "prediction_accuracy": accuracy}
    )
    return V1Config.from_dict(v1_config_dict)


def test_small_run_tracks_layer_kv_and_required_resources(v1_config_dict):
    config = small_config(v1_config_dict)
    result = V1Simulator(config).run()
    per_layer_token = 2 * 1 * 4
    assert result.summary.initial_kv_bytes == 2 * 3 * per_layer_token
    assert result.summary.final_kv_bytes == 5 * 3 * per_layer_token
    assert [state.context_length for state in result.token_states] == [3, 4, 5]
    assert all(
        later.kv_occupancy_bytes > earlier.kv_occupancy_bytes
        for earlier, later in zip(result.token_states, result.token_states[1:])
    )
    assert {event.resource for event in result.events} == {
        OFF_CHIP_DMA,
        ON_CHIP_KV_READ,
        COMPUTE,
    }


def test_prefetch_produces_dma_compute_overlap_and_no_waste_at_accuracy_one(v1_config_dict):
    result = V1Simulator(small_config(v1_config_dict, True, 1.0)).run()
    assert result.summary.dma_compute_overlap_cycles > 0
    assert result.summary.useful_prefetch_bytes > 0
    assert result.summary.wasted_prefetch_bytes == 0
    assert result.summary.observed_prediction_accuracy == 1.0


def test_prefetch_disabled_is_demand_only_baseline(v1_config_dict):
    result = V1Simulator(small_config(v1_config_dict, False, 0.0)).run()
    assert result.summary.useful_prefetch_bytes == 0
    assert result.summary.wasted_prefetch_bytes == 0
    assert result.summary.demand_expert_bytes == result.summary.total_expert_dma_bytes
    expert_size = 3 * 8 * 4
    assert result.summary.demand_expert_bytes == result.summary.cache_misses * expert_size
    assert all(
        event.prefetch_or_demand != "prefetch"
        for event in result.events
        if event.resource == OFF_CHIP_DMA
    )


def test_dma_bytes_and_on_chip_capacity_are_conserved(v1_config_dict):
    config = small_config(v1_config_dict, True, 0.0)
    result = V1Simulator(config).run()
    summary = result.summary
    assert summary.total_expert_dma_bytes == (
        summary.useful_prefetch_bytes
        + summary.wasted_prefetch_bytes
        + summary.demand_expert_bytes
    )
    reserved = (
        config.hardware.fixed_reserved_bytes + config.hardware.expert_workspace_bytes
    )
    assert all(
        state.kv_occupancy_bytes + state.expert_cache_occupancy_bytes + reserved
        <= config.hardware.on_chip_memory_bytes
        for state in result.token_states
    )


def test_same_seed_is_exactly_deterministic(v1_config_dict):
    config = small_config(v1_config_dict, True, 0.8)
    assert V1Simulator(config).run().to_dict() == V1Simulator(config).run().to_dict()
