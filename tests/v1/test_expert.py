from simulator.v1.expert import (
    attention_operations,
    compute_cycles,
    dma_cycles,
    expert_bytes,
    expert_operations,
    get_expert_segments,
    kv_bytes_per_layer_token,
    router_operations,
)


def test_qwen_derived_sizes_and_operations(v1_config):
    model = v1_config.model
    assert expert_bytes(model) == 4_325_376
    assert expert_operations(model) == 17_301_504
    assert router_operations(model) == 245_760
    assert kv_bytes_per_layer_token(model) == 8 * 1024
    assert attention_operations(model, 33) == 33 * 8192


def test_one_expert_is_one_complete_segment(v1_config):
    segments = get_expert_segments(3, 7, v1_config.model)
    assert len(segments) == 1
    assert segments[0].key.layer_id == 3
    assert segments[0].key.expert_id == 7
    assert segments[0].name == "full_expert"
    assert segments[0].bytes == 4_325_376


def test_default_expert_dma_and_compute_cycles(v1_config):
    model = v1_config.model
    hardware = v1_config.hardware
    assert dma_cycles(expert_bytes(model), hardware) == 6_509
    segments = get_expert_segments(0, 0, model)
    segment = segments[0]
    assert compute_cycles(segment.gate_up_operations, hardware) == 3_462
    assert compute_cycles(segment.down_operations, hardware) == 1_732
