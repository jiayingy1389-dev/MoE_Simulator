"""Immutable records shared by V1 components."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


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


@dataclass(frozen=True)
class TokenState:
    token_id: int
    context_length: int
    start_cycle: int
    finish_cycle: int
    kv_occupancy_bytes: int
    expert_cache_capacity_bytes: int
    expert_cache_occupancy_bytes: int
    resident_expert_count: int
    kv_eviction_count: int
    total_eviction_count: int


@dataclass(frozen=True)
class V1Summary:
    total_cycles: int
    initial_kv_bytes: int
    final_kv_bytes: int
    peak_kv_bytes: int
    peak_expert_cache_bytes: int
    expert_cache_hits: int
    expert_cache_misses: int
    expert_cache_hit_rate: float
    expert_evictions: int
    kv_triggered_expert_evictions: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    kv_eviction_count: int
    total_eviction_count: int
    configured_prediction_accuracy: float
    observed_prediction_accuracy: float
    useful_prefetch_bytes: int
    wasted_prefetch_bytes: int
    demand_expert_bytes: int
    total_expert_dma_bytes: int
    total_kv_read_bytes: int
    expert_wait_cycles: int
    dma_compute_overlap_cycles: int
    kv_capacity_exceeded: bool
    status: str


@dataclass(frozen=True)
class V1Result:
    events: List[ResourceEvent]
    token_states: List[TokenState]
    summary: V1Summary

    def to_dict(self) -> Dict[str, Any]:
        events = sorted(
            self.events,
            key=lambda event: (event.start_cycle, event.resource, event.sequence_id),
        )
        return {
            "events": [asdict(event) for event in events],
            "token_states": [asdict(state) for state in self.token_states],
            "summary": asdict(self.summary),
        }
