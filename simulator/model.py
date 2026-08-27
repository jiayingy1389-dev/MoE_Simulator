"""Serializable event and summary records."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


class DictRecord:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimelineEvent(DictRecord):
    event_id: int
    depends_on: Optional[int]
    start_cycle: int
    end_cycle: int
    duration: int
    token_id: Optional[int]
    stage: str
    resource: str
    expert_id: Optional[int]
    tile_id: Optional[int]
    shape: Optional[Dict[str, int]]
    bytes_transferred: int
    operations: int
    on_chip_kv_bytes: int
    off_chip_kv_bytes: int


@dataclass(frozen=True)
class TokenSummary(DictRecord):
    token_id: int
    latency: int
    start_cycle: int
    end_cycle: int


@dataclass(frozen=True)
class SequenceSummary(DictRecord):
    total_cycles: int
    memory_cycles: int
    compute_cycles: int
    nonlinear_cycles: int
    off_chip_kv_read_bytes: int
    off_chip_kv_write_bytes: int
    expert_weight_read_bytes: int
    on_chip_kv_peak_bytes: int
    off_chip_kv_peak_bytes: int
    first_kv_spill_token: Optional[int]


@dataclass(frozen=True)
class SimulationResult(DictRecord):
    timeline: List[TimelineEvent]
    tokens: List[TokenSummary]
    summary: SequenceSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeline": [event.to_dict() for event in self.timeline],
            "tokens": [token.to_dict() for token in self.tokens],
            "summary": self.summary.to_dict(),
        }
