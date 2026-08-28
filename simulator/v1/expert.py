"""Parameter-derived V1 Expert, KV, and timing formulas."""

import math
from typing import List

from .config import V1HardwareConfig, V1ModelConfig
from .model import ExpertKey, WeightSegment


def ceil_bytes(bits: int) -> int:
    return (bits + 7) // 8


def expert_bytes(model: V1ModelConfig) -> int:
    return ceil_bytes(3 * model.hidden_size * model.expert_intermediate_size * model.weight_bits)


def expert_operations(model: V1ModelConfig) -> int:
    return 6 * model.hidden_size * model.expert_intermediate_size


def router_operations(model: V1ModelConfig) -> int:
    return 2 * model.hidden_size * model.num_routed_experts


def kv_bytes_per_layer_token(model: V1ModelConfig) -> int:
    return ceil_bytes(2 * model.num_kv_heads * model.head_dim * model.kv_bits)


def attention_operations(model: V1ModelConfig, context_length: int) -> int:
    return model.attention_base_ops + context_length * model.attention_ops_per_context_token


def dma_cycles(bytes_count: int, hardware: V1HardwareConfig) -> int:
    return hardware.off_chip_dma_startup_cycles + math.ceil(
        bytes_count / hardware.off_chip_bytes_per_cycle
    )


def compute_cycles(operations: int, hardware: V1HardwareConfig) -> int:
    return hardware.compute_startup_cycles + math.ceil(
        operations / hardware.compute_ops_per_cycle
    )


def get_expert_segments(
    layer_id: int, expert_id: int, model: V1ModelConfig
) -> List[WeightSegment]:
    h = model.hidden_size
    f = model.expert_intermediate_size
    return [
        WeightSegment(
            key=ExpertKey(layer_id, expert_id),
            name="full_expert",
            bytes=expert_bytes(model),
            gate_up_operations=4 * h * f,
            down_operations=2 * h * f,
        )
    ]
