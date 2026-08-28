"""Cycle-level V1 MoE cache, prefetch, and resource simulator."""

import math
from typing import Dict, List, Optional, Set

from .cache import CacheEntryState, Eviction, ExpertCache
from .config import V1Config
from .expert import (
    attention_operations,
    compute_cycles,
    expert_bytes,
    get_expert_segments,
    router_operations,
)
from .kv import KVCapacityExceeded, KVManager
from .model import (
    COMPUTE,
    OFF_CHIP_DMA,
    ON_CHIP_KV_READ,
    ExpertKey,
    TokenState,
    V1Result,
    V1Summary,
)
from .predictor import SyntheticPredictor
from .routing import SyntheticRoutingProvider
from .scheduler import DMARequest, ResourceScheduler, overlap_cycles


class V1Simulator:
    """Simulate one request with full-Expert transfers and adjacent-layer prefetch."""

    def __init__(self, config: V1Config) -> None:
        self.config = config
        self.kv = KVManager(config)
        self.initial_kv_bytes = self.kv.total_bytes
        self.expert_size = expert_bytes(config.model)
        self.cache = ExpertCache(
            self.expert_size, config.hardware.expert_workspace_bytes
        )
        self.scheduler = ResourceScheduler(config.hardware)
        self.routing = SyntheticRoutingProvider(
            config.model, config.request.routing_seed
        )
        self.predictor = SyntheticPredictor(
            config.model,
            self.routing,
            config.prefetch.prediction_accuracy,
            config.prefetch.prediction_seed,
        )
        self.token_states: List[TokenState] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.kv_evictions = 0
        self.total_evictions = 0
        self.prediction_total = 0
        self.prediction_correct = 0
        self.useful_prefetch_bytes = 0
        self.wasted_prefetch_bytes = 0
        self.demand_expert_bytes = 0
        self.expert_wait_cycles = 0
        self.completed_loads: Set[ExpertKey] = set()
        self.pending_loads: Dict[ExpertKey, DMARequest] = {}
        self.prefetch_requests: Dict[ExpertKey, DMARequest] = {}
        self.unclassified_prefetch: Dict[ExpertKey, int] = {}
        self.kv_capacity_exceeded = False

    def _classify_evictions(self, evictions: List[Eviction]) -> None:
        for eviction in evictions:
            self.total_evictions += 1
            self.completed_loads.discard(eviction.key)
            if eviction.prefetched and not eviction.used:
                size = self.unclassified_prefetch.pop(eviction.key, 0)
                self.wasted_prefetch_bytes += size

    def _load_start(self, request: DMARequest, now: int) -> bool:
        admission = self.cache.reserve_load(
            request.key, self.kv.cache_capacity_bytes, request.kind, now
        )
        self._classify_evictions(list(admission.evictions))
        return admission.status in ("cache", "workspace", "existing")

    def _load_complete(self, request: DMARequest, now: int) -> None:
        self.cache.mark_resident(request.key)
        self.completed_loads.add(request.key)
        self.pending_loads.pop(request.key, None)
        if request.kind == "demand":
            self.demand_expert_bytes += request.bytes
        else:
            self.unclassified_prefetch[request.key] = request.bytes
            self.prefetch_requests.pop(request.key, None)

    def _enqueue_load(
        self,
        key: ExpertKey,
        kind: str,
        now: int,
        token_id: int,
        prediction_correct: Optional[bool] = None,
    ) -> DMARequest:
        request = self.scheduler.enqueue_dma(
            now,
            self.expert_size,
            key,
            kind,
            token_id,
            key.layer_id,
            self._load_start,
            self._load_complete,
        )
        request.prediction_correct = prediction_correct
        self.completed_loads.discard(key)
        self.pending_loads[key] = request
        if kind == "prefetch":
            self.prefetch_requests[key] = request
        return request

    def _wait_for_load(self, key: ExpertKey, now: int) -> int:
        start = now
        while key not in self.completed_loads:
            self.scheduler.advance_dma(now)
            if key in self.completed_loads:
                break
            active = self.scheduler.active_dma
            if active is None:
                event = self.scheduler.start_next_dma(now)
                if event is None:
                    raise RuntimeError("Expert load was skipped or blocked: {}".format(key))
                active = self.scheduler.active_dma
            now = active.event.end_cycle
            self.scheduler.advance_dma(now)
        self.expert_wait_cycles += now - start
        return now

    def _make_room_for_kv(self, now: int) -> int:
        required_capacity = self.kv.cache_capacity_bytes - self.kv.bytes_per_layer_token
        if required_capacity < 0:
            raise KVCapacityExceeded()
        while True:
            evictions, blocked = self.cache.evict_for_capacity(required_capacity)
            self._classify_evictions(evictions)
            if not blocked:
                return now
            active = self.scheduler.active_dma
            if active is None:
                raise KVCapacityExceeded()
            now = max(now, active.event.end_cycle)
            self.scheduler.advance_dma(now)

    def _resolve_current_predictions(self, actual: Set[ExpertKey]) -> None:
        for key, request in list(self.prefetch_requests.items()):
            if key.layer_id not in {key_.layer_id for key_ in actual}:
                continue
            if key not in actual:
                if self.scheduler.cancel_queued_prefetch(key):
                    self.prefetch_requests.pop(key, None)
                    self.pending_loads.pop(key, None)
                else:
                    self.scheduler.mark_inflight_prefetch_wrong(key)

    def _enqueue_prediction(self, token_id: int, layer_id: int, now: int) -> None:
        if not self.config.prefetch.prefetch_enabled:
            return
        if layer_id + 1 >= self.config.model.num_layers:
            return
        prediction = self.predictor.predict(token_id, layer_id, layer_id + 1)
        self.prediction_total += len(prediction.expert_ids)
        self.prediction_correct += prediction.correct_count
        truth = set(prediction.true_expert_ids)
        for expert_id in prediction.expert_ids:
            key = ExpertKey(layer_id + 1, expert_id)
            if key in self.cache.entries or key in self.prefetch_requests:
                continue
            self._enqueue_load(key, "prefetch", now, token_id, expert_id in truth)

    def _use_expert(self, key: ExpertKey, now: int, token_id: int) -> int:
        entry = self.cache.entries.get(key)
        pending = self.pending_loads.get(key)
        if entry is not None and entry.state == CacheEntryState.RESIDENT:
            pass
        elif pending is not None or (
            entry is not None and entry.state == CacheEntryState.LOADING
        ):
            now = self._wait_for_load(key, now)
        else:
            self._enqueue_load(key, "demand", now, token_id)
            now = self._wait_for_load(key, now)

        prefetched = self.unclassified_prefetch.pop(key, 0)
        self.useful_prefetch_bytes += prefetched
        self.cache.begin_compute(key, now)
        segment = get_expert_segments(
            key.layer_id, key.expert_id, self.config.model
        )[0]
        gate = self.scheduler.schedule_compute(
            now,
            compute_cycles(segment.gate_up_operations, self.config.hardware),
            "EXPERT_GATE_UP",
            token_id,
            key.layer_id,
            key.expert_id,
            segment.gate_up_operations,
        )
        down = self.scheduler.schedule_compute(
            gate.end_cycle,
            compute_cycles(segment.down_operations, self.config.hardware),
            "EXPERT_DOWN",
            token_id,
            key.layer_id,
            key.expert_id,
            segment.down_operations,
        )
        self.scheduler.advance_dma(down.end_cycle)
        self.cache.finish_compute(key)
        self.completed_loads.discard(key)
        return down.end_cycle

    def _run_layer(self, token_id: int, layer_id: int, now: int) -> int:
        now = self._make_room_for_kv(now)
        self.kv.add_layer_token(layer_id)
        context_length = self.config.request.prompt_tokens + token_id + 1
        kv_bytes = context_length * self.kv.bytes_per_layer_token
        kv_read = self.scheduler.schedule_kv_read(
            now,
            math.ceil(kv_bytes / self.config.hardware.on_chip_read_bytes_per_cycle),
            kv_bytes,
            token_id,
            layer_id,
        )
        attention_ops = attention_operations(self.config.model, context_length)
        attention = self.scheduler.schedule_compute(
            kv_read.end_cycle,
            compute_cycles(attention_ops, self.config.hardware),
            "ATTENTION",
            token_id,
            layer_id,
            operations=attention_ops,
        )
        route_ops = router_operations(self.config.model)
        router = self.scheduler.schedule_compute(
            attention.end_cycle,
            compute_cycles(route_ops, self.config.hardware),
            "ROUTER",
            token_id,
            layer_id,
            operations=route_ops,
        )
        now = router.end_cycle + self.config.prefetch.predictor_latency_cycles
        actual = {
            ExpertKey(layer_id, expert_id)
            for expert_id in self.routing.get_active_experts(token_id, layer_id)
        }
        self._resolve_current_predictions(actual)
        for key in actual:
            entry = self.cache.entries.get(key)
            pending = self.pending_loads.get(key)
            if entry is not None and (
                entry.state == CacheEntryState.RESIDENT or entry.prefetched
            ):
                self.cache_hits += 1
            elif pending is not None and pending.kind == "prefetch":
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            if pending is None and entry is None:
                self._enqueue_load(key, "demand", now, token_id)
        self._enqueue_prediction(token_id, layer_id, now)
        self.scheduler.advance_dma(now)
        for key in sorted(actual, key=lambda item: item.expert_id):
            now = self._use_expert(key, now, token_id)
        return now

    def _summary(self) -> V1Summary:
        events = self.scheduler.events
        dma_events = [event for event in events if event.resource == OFF_CHIP_DMA]
        compute_events = [event for event in events if event.resource == COMPUTE]
        kv_events = [event for event in events if event.resource == ON_CHIP_KV_READ]
        total_accesses = self.cache_hits + self.cache_misses
        return V1Summary(
            total_cycles=max((event.end_cycle for event in events), default=0),
            initial_kv_bytes=self.initial_kv_bytes,
            final_kv_bytes=self.kv.total_bytes,
            peak_kv_bytes=self.kv.peak_bytes,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cache_hit_rate=(self.cache_hits / total_accesses if total_accesses else 0.0),
            kv_eviction_count=self.kv_evictions,
            total_eviction_count=self.total_evictions,
            configured_prediction_accuracy=self.config.prefetch.prediction_accuracy,
            observed_prediction_accuracy=(
                self.prediction_correct / self.prediction_total
                if self.prediction_total else 0.0
            ),
            useful_prefetch_bytes=self.useful_prefetch_bytes,
            wasted_prefetch_bytes=self.wasted_prefetch_bytes,
            demand_expert_bytes=self.demand_expert_bytes,
            total_expert_dma_bytes=sum(event.bytes for event in dma_events),
            total_kv_read_bytes=sum(event.bytes for event in kv_events),
            expert_wait_cycles=self.expert_wait_cycles,
            dma_compute_overlap_cycles=overlap_cycles(dma_events, compute_events),
            kv_capacity_exceeded=self.kv_capacity_exceeded,
            status="KV_CAPACITY_EXCEEDED" if self.kv_capacity_exceeded else "OK",
        )

    def run(self) -> V1Result:
        now = 0
        try:
            for token_id in range(self.config.request.decode_tokens):
                token_start = now
                evictions_before = self.total_evictions
                for layer_id in range(self.config.model.num_layers):
                    now = self._run_layer(token_id, layer_id, now)
                self.scheduler.advance_dma(now)
                self.token_states.append(
                    TokenState(
                        token_id=token_id,
                        context_length=self.config.request.prompt_tokens + token_id + 1,
                        start_cycle=token_start,
                        finish_cycle=now,
                        kv_occupancy_bytes=self.kv.total_bytes,
                        expert_cache_capacity_bytes=self.kv.cache_capacity_bytes,
                        expert_cache_occupancy_bytes=self.cache.occupancy_bytes,
                        resident_expert_count=self.cache.resident_count,
                        kv_eviction_count=self.total_evictions - evictions_before,
                        total_eviction_count=self.total_evictions,
                    )
                )
        except KVCapacityExceeded:
            self.kv_capacity_exceeded = True

        while self.scheduler.active_dma is not None:
            self.scheduler.advance_dma(self.scheduler.active_dma.event.end_cycle)
        for size in self.unclassified_prefetch.values():
            self.wasted_prefetch_bytes += size
        self.unclassified_prefetch.clear()
        return V1Result(self.scheduler.events, self.token_states, self._summary())
