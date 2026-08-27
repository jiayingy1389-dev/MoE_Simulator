"""Strictly serialized V0 event timeline."""

from typing import List, Optional

from .compute import compute_cycles, dma_cycles
from .config import HardwareConfig
from .model import TimelineEvent


class Timeline:
    def __init__(self, hardware: HardwareConfig) -> None:
        self.hardware = hardware
        self.events: List[TimelineEvent] = []
        self.current_cycle = 0

    def add_dma(
        self,
        *,
        token_id: Optional[int],
        stage: str,
        bytes_transferred: int,
        on_chip_kv_bytes: int,
        off_chip_kv_bytes: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None
    ) -> TimelineEvent:
        if bytes_transferred <= 0:
            raise ValueError("DMA bytes_transferred must be positive")
        return self._append(
            duration=dma_cycles(bytes_transferred, self.hardware),
            token_id=token_id,
            stage=stage,
            resource="memory",
            expert_id=expert_id,
            tile_id=tile_id,
            bytes_transferred=bytes_transferred,
            operations=0,
            on_chip_kv_bytes=on_chip_kv_bytes,
            off_chip_kv_bytes=off_chip_kv_bytes,
        )

    def add_compute(
        self,
        *,
        token_id: Optional[int],
        stage: str,
        operations: int,
        on_chip_kv_bytes: int,
        off_chip_kv_bytes: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None
    ) -> TimelineEvent:
        if operations <= 0:
            raise ValueError("compute operations must be positive")
        return self._append(
            duration=compute_cycles(operations, self.hardware),
            token_id=token_id,
            stage=stage,
            resource="compute",
            expert_id=expert_id,
            tile_id=tile_id,
            bytes_transferred=0,
            operations=operations,
            on_chip_kv_bytes=on_chip_kv_bytes,
            off_chip_kv_bytes=off_chip_kv_bytes,
        )

    def add_nonlinear(
        self,
        *,
        token_id: Optional[int],
        stage: str,
        on_chip_kv_bytes: int,
        off_chip_kv_bytes: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None
    ) -> TimelineEvent:
        return self._append(
            duration=self.hardware.nonlinear_cycles_per_tile,
            token_id=token_id,
            stage=stage,
            resource="compute",
            expert_id=expert_id,
            tile_id=tile_id,
            bytes_transferred=0,
            operations=0,
            on_chip_kv_bytes=on_chip_kv_bytes,
            off_chip_kv_bytes=off_chip_kv_bytes,
        )

    def add_state(
        self,
        *,
        token_id: Optional[int],
        stage: str,
        on_chip_kv_bytes: int,
        off_chip_kv_bytes: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None
    ) -> TimelineEvent:
        return self._append(
            duration=0,
            token_id=token_id,
            stage=stage,
            resource="state",
            expert_id=expert_id,
            tile_id=tile_id,
            bytes_transferred=0,
            operations=0,
            on_chip_kv_bytes=on_chip_kv_bytes,
            off_chip_kv_bytes=off_chip_kv_bytes,
        )

    def _append(
        self,
        *,
        duration: int,
        token_id: Optional[int],
        stage: str,
        resource: str,
        expert_id: Optional[int],
        tile_id: Optional[int],
        bytes_transferred: int,
        operations: int,
        on_chip_kv_bytes: int,
        off_chip_kv_bytes: int
    ) -> TimelineEvent:
        event_id = len(self.events)
        start_cycle = self.current_cycle
        event = TimelineEvent(
            event_id=event_id,
            depends_on=event_id - 1 if event_id else None,
            start_cycle=start_cycle,
            end_cycle=start_cycle + duration,
            duration=duration,
            token_id=token_id,
            stage=stage,
            resource=resource,
            expert_id=expert_id,
            tile_id=tile_id,
            bytes_transferred=bytes_transferred,
            operations=operations,
            on_chip_kv_bytes=on_chip_kv_bytes,
            off_chip_kv_bytes=off_chip_kv_bytes,
        )
        self.events.append(event)
        self.current_cycle = event.end_cycle
        return event

    @property
    def total_cycles(self) -> int:
        return self.current_cycle

    @property
    def memory_cycles(self) -> int:
        return sum(event.duration for event in self.events if event.resource == "memory")

    @property
    def compute_cycles(self) -> int:
        return sum(event.duration for event in self.events if event.resource == "compute")

    @property
    def nonlinear_cycles(self) -> int:
        return sum(event.duration for event in self.events if event.stage == "expert_nonlinear")
