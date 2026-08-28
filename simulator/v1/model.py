"""Immutable records shared by V1 components."""

from dataclasses import dataclass


OFF_CHIP_DMA = "OFF_CHIP_DMA"
ON_CHIP_KV_READ = "ON_CHIP_KV_READ"
COMPUTE = "COMPUTE"


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int


@dataclass(frozen=True)
class WeightSegment:
    key: ExpertKey
    name: str
    bytes: int
    gate_up_operations: int
    down_operations: int
