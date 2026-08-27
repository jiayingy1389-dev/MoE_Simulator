"""Single-layer, decode-only MoE V0 orchestration."""

from typing import List, Optional

from .compute import ceil_bytes, expert_tile_metrics, f_tiles
from .config import SimulationConfig
from .memory import KVMemory
from .model import SequenceSummary, SimulationResult, TokenSummary
from .timeline import Timeline


class MoESimulator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self) -> SimulationResult:
        config = self.config
        hardware = config.hardware
        model = config.model
        request = config.request
        kv_budget = (
            hardware.on_chip_capacity_bytes
            - hardware.fixed_reserved_bytes
            - hardware.workspace_bytes
        )
        kv_bytes_per_token = ceil_bytes(
            2 * model.num_kv_heads * model.head_dim * model.kv_bits
        )
        memory = KVMemory(kv_budget, kv_bytes_per_token)
        for _ in range(request.initial_prompt_length):
            memory.place_prompt_token()

        timeline = Timeline(hardware)
        token_summaries: List[TokenSummary] = []
        first_spill: Optional[int] = None

        for token_id, selected_experts in enumerate(request.routing_trace):
            token_start = timeline.current_cycle
            self._add_compute(
                timeline,
                memory,
                token_id,
                "qkv_attention_prepare",
                model.qkv_attention_prepare_ops,
            )

            placement = memory.place_decode_token(token_id)
            self._add_state(timeline, memory, token_id, "place_token_kv")
            if placement.location == "off_chip":
                if first_spill is None:
                    first_spill = token_id
                self._add_dma(
                    timeline,
                    memory,
                    token_id,
                    "kv_write",
                    placement.bytes_count,
                )

            off_chip_kv = memory.off_chip_attention_bytes()
            if off_chip_kv:
                self._add_dma(
                    timeline,
                    memory,
                    token_id,
                    "attention_kv_read",
                    off_chip_kv,
                )

            context_length = request.initial_prompt_length + token_id + 1
            attention_ops = (
                model.attention_base_ops
                + context_length * model.attention_ops_per_context_token
            )
            self._add_compute(
                timeline,
                memory,
                token_id,
                "attention_compute",
                attention_ops,
            )
            self._add_compute(
                timeline,
                memory,
                token_id,
                "router_compute",
                2 * model.H * model.E,
            )

            for expert_id in selected_experts:
                self._run_expert(timeline, memory, token_id, expert_id)

            self._add_compute(
                timeline,
                memory,
                token_id,
                "topk_merge",
                model.topk_merge_ops,
            )
            completion = self._add_state(timeline, memory, token_id, "token_complete")
            token_summaries.append(
                TokenSummary(
                    token_id=token_id,
                    latency=completion.end_cycle - token_start,
                    start_cycle=token_start,
                    end_cycle=completion.end_cycle,
                )
            )

        memory.release()
        self._add_state(timeline, memory, None, "release_request_kv")

        summary = SequenceSummary(
            total_cycles=timeline.total_cycles,
            memory_cycles=timeline.memory_cycles,
            compute_cycles=timeline.compute_cycles,
            nonlinear_cycles=timeline.nonlinear_cycles,
            off_chip_kv_read_bytes=self._bytes_for_stage(timeline, "attention_kv_read"),
            off_chip_kv_write_bytes=self._bytes_for_stage(timeline, "kv_write"),
            expert_weight_read_bytes=sum(
                event.bytes_transferred
                for event in timeline.events
                if event.stage in ("expert_gu_weight_read", "expert_down_weight_read")
            ),
            on_chip_kv_peak_bytes=memory.on_chip_peak_bytes,
            off_chip_kv_peak_bytes=memory.off_chip_peak_bytes,
            first_kv_spill_token=first_spill,
        )
        return SimulationResult(timeline.events, token_summaries, summary)

    def _run_expert(
        self,
        timeline: Timeline,
        memory: KVMemory,
        token_id: int,
        expert_id: int,
    ) -> None:
        model = self.config.model
        self._add_state(
            timeline,
            memory,
            token_id,
            "expert_partial_sum_allocate",
            expert_id=expert_id,
        )
        for tile_id, tile_f in enumerate(f_tiles(model.F, model.f_tile_size)):
            metrics = expert_tile_metrics(model, tile_f)
            self._add_dma(
                timeline,
                memory,
                token_id,
                "expert_gu_weight_read",
                metrics.gu_weight_bytes,
                expert_id,
                tile_id,
            )
            self._add_compute(
                timeline,
                memory,
                token_id,
                "expert_gu_compute",
                metrics.gu_operations,
                expert_id,
                tile_id,
            )
            timeline.add_nonlinear(
                token_id=token_id,
                stage="expert_nonlinear",
                expert_id=expert_id,
                tile_id=tile_id,
                on_chip_kv_bytes=memory.on_chip_bytes,
                off_chip_kv_bytes=memory.off_chip_bytes,
            )
            self._add_dma(
                timeline,
                memory,
                token_id,
                "expert_down_weight_read",
                metrics.down_weight_bytes,
                expert_id,
                tile_id,
            )
            self._add_compute(
                timeline,
                memory,
                token_id,
                "expert_down_compute",
                metrics.down_operations,
                expert_id,
                tile_id,
            )
        self._add_state(
            timeline,
            memory,
            token_id,
            "expert_release",
            expert_id=expert_id,
        )

    @staticmethod
    def _add_dma(
        timeline: Timeline,
        memory: KVMemory,
        token_id: int,
        stage: str,
        bytes_transferred: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None,
    ):
        return timeline.add_dma(
            token_id=token_id,
            stage=stage,
            bytes_transferred=bytes_transferred,
            expert_id=expert_id,
            tile_id=tile_id,
            on_chip_kv_bytes=memory.on_chip_bytes,
            off_chip_kv_bytes=memory.off_chip_bytes,
        )

    @staticmethod
    def _add_compute(
        timeline: Timeline,
        memory: KVMemory,
        token_id: int,
        stage: str,
        operations: int,
        expert_id: Optional[int] = None,
        tile_id: Optional[int] = None,
    ):
        return timeline.add_compute(
            token_id=token_id,
            stage=stage,
            operations=operations,
            expert_id=expert_id,
            tile_id=tile_id,
            on_chip_kv_bytes=memory.on_chip_bytes,
            off_chip_kv_bytes=memory.off_chip_bytes,
        )

    @staticmethod
    def _add_state(
        timeline: Timeline,
        memory: KVMemory,
        token_id: Optional[int],
        stage: str,
        expert_id: Optional[int] = None,
    ):
        return timeline.add_state(
            token_id=token_id,
            stage=stage,
            expert_id=expert_id,
            on_chip_kv_bytes=memory.on_chip_bytes,
            off_chip_kv_bytes=memory.off_chip_bytes,
        )

    @staticmethod
    def _bytes_for_stage(timeline: Timeline, stage: str) -> int:
        return sum(
            event.bytes_transferred for event in timeline.events if event.stage == stage
        )
