"""Replaceable routing interface and deterministic synthetic routing."""

import random
from abc import ABC, abstractmethod
from typing import List

from .config import V1ModelConfig


MASK64 = (1 << 64) - 1


def stable_seed(base_seed: int, *components: int) -> int:
    value = base_seed & MASK64
    for component in components:
        value ^= (component + 0x9E3779B97F4A7C15) & MASK64
        value = (value * 0xBF58476D1CE4E5B9) & MASK64
        value ^= value >> 27
    return value & MASK64


class RoutingProvider(ABC):
    @abstractmethod
    def get_active_experts(self, token_id: int, layer_id: int) -> List[int]:
        raise NotImplementedError


class SyntheticRoutingProvider(RoutingProvider):
    def __init__(self, model: V1ModelConfig, seed: int) -> None:
        self.model = model
        self.seed = seed

    def get_active_experts(self, token_id: int, layer_id: int) -> List[int]:
        rng = random.Random(stable_seed(self.seed, 1, token_id, layer_id))
        return rng.sample(range(self.model.num_routed_experts), self.model.top_k)
