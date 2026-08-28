"""Immutable records shared by V1 components."""

from dataclasses import dataclass
from typing import Optional


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


@dataclass
class ResourceEvent:
    sequence_id: int
    start_cycle: int
    end_cycle: int
    resource: str
    operation: str
    token_id: int
    layer_id: int
    expert_id: Optional[int]
    bytes: int
    operations: int
    prefetch_or_demand: Optional[str]
    prediction_correct: Optional[bool]
