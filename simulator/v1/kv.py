"""Per-layer, permanently on-chip KV accounting for V1."""

from typing import List

from .config import V1Config
from .expert import kv_bytes_per_layer_token


class KVCapacityExceeded(RuntimeError):
    def __init__(self) -> None:
        super().__init__("KV_CAPACITY_EXCEEDED")


class KVManager:
    def __init__(self, config: V1Config) -> None:
        self.config = config
        self.bytes_per_layer_token = kv_bytes_per_layer_token(config.model)
        initial_per_layer = config.request.prompt_tokens * self.bytes_per_layer_token
        self._layer_bytes: List[int] = [initial_per_layer] * config.model.num_layers
        self.peak_bytes = self.total_bytes

    @property
    def total_bytes(self) -> int:
        return sum(self._layer_bytes)

    def layer_bytes(self, layer_id: int) -> int:
        return self._layer_bytes[layer_id]

    @property
    def cache_capacity_bytes(self) -> int:
        hardware = self.config.hardware
        return (
            hardware.on_chip_memory_bytes
            - hardware.fixed_reserved_bytes
            - hardware.expert_workspace_bytes
            - self.total_bytes
        )

    def add_layer_token(self, layer_id: int) -> int:
        if self.cache_capacity_bytes < self.bytes_per_layer_token:
            raise KVCapacityExceeded()
        self._layer_bytes[layer_id] += self.bytes_per_layer_token
        self.peak_bytes = max(self.peak_bytes, self.total_bytes)
        return self.bytes_per_layer_token
