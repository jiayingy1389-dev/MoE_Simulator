import copy

import pytest


@pytest.fixture
def v1_config_dict():
    return copy.deepcopy(
        {
            "model": {
                "model_name": "qwen1.5_moe_a2.7b_routed_only",
                "num_layers": 24,
                "total_parameters": 14.3e9,
                "activated_parameters": 2.7e9,
                "hidden_size": 2048,
                "expert_intermediate_size": 1408,
                "num_routed_experts": 60,
                "top_k": 4,
                "num_kv_heads": 16,
                "head_dim": 128,
                "weight_bits": 4,
                "kv_bits": 16,
                "batch_size": 1,
                "num_requests": 1,
                "attention_base_ops": 0,
                "attention_ops_per_context_token": 8192,
            },
            "request": {
                "prompt_tokens": 32,
                "decode_tokens": 96,
                "routing_source": "synthetic",
                "routing_seed": 2025,
            },
            "hardware": {
                "clock_frequency_mhz": 300,
                "on_chip_memory_mib": 32,
                "fixed_reserved_mib": 3,
                "expert_workspace_mib": 5,
                "off_chip_bandwidth_gbps": 200,
                "off_chip_dma_startup_cycles": 20,
                "on_chip_read_bytes_per_cycle": 4096,
                "compute_tops": 1.0,
                "compute_ops_per_cycle": 3333,
                "compute_startup_cycles": 1,
            },
            "prefetch": {
                "prefetch_enabled": True,
                "prefetch_distance": 1,
                "prediction_accuracy": 0.8,
                "predictor_latency_cycles": 0,
                "prediction_seed": 1234,
            },
        }
    )


@pytest.fixture
def v1_config(v1_config_dict):
    from simulator.v1.config import V1Config

    return V1Config.from_dict(v1_config_dict)
