"""Independent resource calendars and prioritized non-preemptive DMA."""

from dataclasses import dataclass
from typing import Callable, List, Optional

from .config import V1HardwareConfig
from .expert import dma_cycles
from .model import COMPUTE, OFF_CHIP_DMA, ON_CHIP_KV_READ, ExpertKey, ResourceEvent


StartCallback = Callable[["DMARequest", int], bool]
CompleteCallback = Callable[["DMARequest", int], None]


@dataclass
class DMARequest:
    sequence_id: int
    enqueue_cycle: int
    bytes: int
    key: ExpertKey
    kind: str
    token_id: int
    layer_id: int
    on_start: Optional[StartCallback] = None
    on_complete: Optional[CompleteCallback] = None
    event: Optional[ResourceEvent] = None
    prediction_correct: Optional[bool] = None


@dataclass
class ActiveDMA:
    request: DMARequest
    event: ResourceEvent


class ResourceScheduler:
    def __init__(self, hardware: V1HardwareConfig) -> None:
        self.hardware = hardware
        self.compute_available = 0
        self.kv_read_available = 0
        self.dma_available = 0
        self._sequence = 0
        self.demand_queue: List[DMARequest] = []
        self.prefetch_queue: List[DMARequest] = []
        self.active_dma: Optional[ActiveDMA] = None
        self.events: List[ResourceEvent] = []

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence += 1
        return value

    def schedule_compute(self, earliest: int, duration: int, operation: str,
                         token_id: int, layer_id: int,
                         expert_id: Optional[int] = None,
                         operations: int = 0) -> ResourceEvent:
        start = max(earliest, self.compute_available)
        event = ResourceEvent(
            self._next_sequence(), start, start + duration, COMPUTE, operation,
            token_id, layer_id, expert_id, 0, operations, None, None,
        )
        self.compute_available = event.end_cycle
        self.events.append(event)
        return event

    def schedule_kv_read(self, earliest: int, duration: int, bytes_count: int,
                         token_id: int, layer_id: int) -> ResourceEvent:
        start = max(earliest, self.kv_read_available)
        event = ResourceEvent(
            self._next_sequence(), start, start + duration, ON_CHIP_KV_READ,
            "KV_READ", token_id, layer_id, None, bytes_count, 0, None, None,
        )
        self.kv_read_available = event.end_cycle
        self.events.append(event)
        return event

    def enqueue_dma(self, enqueue_cycle: int, bytes_count: int, key: ExpertKey,
                    kind: str, token_id: int, layer_id: int,
                    on_start: Optional[StartCallback] = None,
                    on_complete: Optional[CompleteCallback] = None) -> DMARequest:
        request = DMARequest(
            self._next_sequence(), enqueue_cycle, bytes_count, key, kind,
            token_id, layer_id, on_start, on_complete,
        )
        (self.demand_queue if kind == "demand" else self.prefetch_queue).append(request)
        return request

    def start_next_dma(self, now: int) -> Optional[ResourceEvent]:
        if self.active_dma is not None:
            return self.active_dma.event
        while self.demand_queue or self.prefetch_queue:
            queue = self.demand_queue if self.demand_queue else self.prefetch_queue
            request = queue.pop(0)
            start = max(now, self.dma_available, request.enqueue_cycle)
            if request.on_start is not None and not request.on_start(request, start):
                continue
            duration = dma_cycles(request.bytes, self.hardware)
            event = ResourceEvent(
                request.sequence_id, start, start + duration, OFF_CHIP_DMA,
                "EXPERT_LOAD", request.token_id, request.layer_id,
                request.key.expert_id, request.bytes, 0, request.kind,
                request.prediction_correct,
            )
            request.event = event
            self.active_dma = ActiveDMA(request, event)
            self.dma_available = event.end_cycle
            self.events.append(event)
            return event
        return None

    def advance_dma(self, now: int) -> List[ResourceEvent]:
        completed: List[ResourceEvent] = []
        if self.active_dma is None:
            self.start_next_dma(now)
        while self.active_dma is not None and self.active_dma.event.end_cycle <= now:
            active = self.active_dma
            self.active_dma = None
            if active.request.on_complete is not None:
                active.request.on_complete(active.request, active.event.end_cycle)
            completed.append(active.event)
            self.start_next_dma(active.event.end_cycle)
        return completed

    def cancel_queued_prefetch(self, key: ExpertKey) -> bool:
        for index, request in enumerate(self.prefetch_queue):
            if request.key == key:
                self.prefetch_queue.pop(index)
                return True
        return False

    def mark_inflight_prefetch_wrong(self, key: ExpertKey) -> bool:
        if (self.active_dma is not None
                and self.active_dma.request.kind == "prefetch"
                and self.active_dma.request.key == key):
            self.active_dma.request.prediction_correct = False
            self.active_dma.event.prediction_correct = False
            return True
        return False


def overlap_cycles(left_events: List[ResourceEvent],
                   right_events: List[ResourceEvent]) -> int:
    left = sorted((event.start_cycle, event.end_cycle) for event in left_events)
    right = sorted((event.start_cycle, event.end_cycle) for event in right_events)
    total = 0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total
