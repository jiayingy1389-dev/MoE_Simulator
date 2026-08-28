"""Protected-entry LRU cache for complete routed Experts."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .model import ExpertKey


class CacheEntryState(str, Enum):
    LOADING = "LOADING"
    RESIDENT = "RESIDENT"
    IN_COMPUTE = "IN_COMPUTE"


@dataclass
class CacheEntry:
    key: ExpertKey
    bytes: int
    state: CacheEntryState
    last_used_time: int
    admission_order: int
    prefetched: bool
    used: bool = False


@dataclass(frozen=True)
class Eviction:
    key: ExpertKey
    bytes: int
    prefetched: bool
    used: bool


@dataclass(frozen=True)
class AdmissionResult:
    status: str
    evictions: Tuple[Eviction, ...] = ()


class ExpertCache:
    def __init__(self, expert_bytes: int, workspace_bytes: int) -> None:
        self.expert_bytes = expert_bytes
        self.workspace_bytes = workspace_bytes
        self.entries: Dict[ExpertKey, CacheEntry] = {}
        self.workspace_key: Optional[ExpertKey] = None
        self._admission_counter = 0

    @property
    def occupancy_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries.values())

    @property
    def resident_count(self) -> int:
        return sum(
            entry.state == CacheEntryState.RESIDENT for entry in self.entries.values()
        )

    def reserve_load(
        self,
        key: ExpertKey,
        capacity_bytes: int,
        request_kind: str,
        now: int,
    ) -> AdmissionResult:
        if key in self.entries or key == self.workspace_key:
            return AdmissionResult("existing")
        if capacity_bytes < self.expert_bytes:
            if request_kind == "demand" and self.workspace_key is None:
                if self.workspace_bytes < self.expert_bytes:
                    return AdmissionResult("blocked")
                self.workspace_key = key
                return AdmissionResult("workspace")
            return AdmissionResult("skipped" if request_kind == "prefetch" else "blocked")

        evictions: List[Eviction] = []
        while self.occupancy_bytes + self.expert_bytes > capacity_bytes:
            candidate = self.select_lru()
            if candidate is None:
                return AdmissionResult(
                    "skipped" if request_kind == "prefetch" else "blocked",
                    tuple(evictions),
                )
            evictions.append(self._evict(candidate))

        self._admission_counter += 1
        self.entries[key] = CacheEntry(
            key=key,
            bytes=self.expert_bytes,
            state=CacheEntryState.LOADING,
            last_used_time=-1,
            admission_order=self._admission_counter,
            prefetched=request_kind == "prefetch",
        )
        return AdmissionResult("cache", tuple(evictions))

    def mark_resident(self, key: ExpertKey) -> None:
        if key in self.entries:
            self.entries[key].state = CacheEntryState.RESIDENT

    def begin_compute(self, key: ExpertKey, now: int) -> None:
        if key in self.entries:
            entry = self.entries[key]
            entry.state = CacheEntryState.IN_COMPUTE
            entry.last_used_time = now
            entry.used = True
        elif key != self.workspace_key:
            raise KeyError(key)

    def finish_compute(self, key: ExpertKey) -> None:
        if key in self.entries:
            self.entries[key].state = CacheEntryState.RESIDENT
        elif key == self.workspace_key:
            self.workspace_key = None
        else:
            raise KeyError(key)

    def touch(self, key: ExpertKey, now: int) -> None:
        entry = self.entries[key]
        entry.last_used_time = now
        entry.used = True

    def select_lru(self) -> Optional[ExpertKey]:
        candidates = [
            entry
            for entry in self.entries.values()
            if entry.state == CacheEntryState.RESIDENT
        ]
        if not candidates:
            return None
        selected = min(
            candidates, key=lambda entry: (entry.last_used_time, entry.admission_order)
        )
        return selected.key

    def _evict(self, key: ExpertKey) -> Eviction:
        entry = self.entries.pop(key)
        return Eviction(entry.key, entry.bytes, entry.prefetched, entry.used)

    def evict_for_capacity(self, capacity_bytes: int) -> Tuple[List[Eviction], bool]:
        evictions: List[Eviction] = []
        while self.occupancy_bytes > capacity_bytes:
            candidate = self.select_lru()
            if candidate is None:
                return evictions, True
            evictions.append(self._evict(candidate))
        return evictions, False

    def discard(self, key: ExpertKey) -> Eviction:
        return self._evict(key)
