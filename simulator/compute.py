"""Pure byte, operation, tile, and cycle formulas."""

from dataclasses import dataclass
from typing import List

from .config import HardwareConfig, ModelConfig


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def ceil_bytes(bits: int) -> int:
    return ceil_div(bits, 8)


def f_tiles(total_f: int, tile_size: int) -> List[int]:
    if total_f <= 0 or tile_size <= 0:
        raise ValueError("F and tile_size must be positive")
    full_tiles, remainder = divmod(total_f, tile_size)
    tiles = [tile_size] * full_tiles
    if remainder:
        tiles.append(remainder)
    return tiles


@dataclass(frozen=True)
class ExpertTileMetrics:
    tile_f: int
    gu_weight_bytes: int
    down_weight_bytes: int
    gu_operations: int
    down_operations: int


def expert_tile_metrics(model: ModelConfig, tile_f: int) -> ExpertTileMetrics:
    return ExpertTileMetrics(
        tile_f=tile_f,
        gu_weight_bytes=ceil_bytes(2 * model.H * tile_f * model.weight_bits),
        down_weight_bytes=ceil_bytes(model.H * tile_f * model.weight_bits),
        gu_operations=4 * model.H * tile_f,
        down_operations=2 * model.H * tile_f,
    )


def dma_cycles(bytes_count: int, hardware: HardwareConfig) -> int:
    return hardware.dma_startup_cycles + ceil_div(bytes_count, hardware.off_chip_bytes_per_cycle)


def compute_cycles(operations: int, hardware: HardwareConfig) -> int:
    return hardware.compute_startup_cycles + ceil_div(operations, hardware.compute_ops_per_cycle)


def required_workspace_bytes(model: ModelConfig) -> int:
    partial_sum = ceil_bytes(model.H * model.accumulator_bits)
    required = 0
    for tile_f in f_tiles(model.F, model.f_tile_size):
        metrics = expert_tile_metrics(model, tile_f)
        gu_intermediate = ceil_bytes(2 * tile_f * model.activation_bits)
        nonlinear_output = ceil_bytes(tile_f * model.activation_bits)
        required = max(
            required,
            partial_sum + metrics.gu_weight_bytes + gu_intermediate,
            partial_sum + metrics.down_weight_bytes + nonlinear_output,
        )
    return required
