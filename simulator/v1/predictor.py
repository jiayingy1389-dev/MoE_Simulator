"""Replaceable predictor and controlled synthetic prediction."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from .config import V1ModelConfig
from .routing import RoutingProvider, stable_seed


@dataclass(frozen=True)
class Prediction:
    expert_ids: Tuple[int, ...]
    true_expert_ids: Tuple[int, ...]
    correct_count: int


class Predictor(ABC):
    @abstractmethod
    def predict(self, token_id: int, source_layer: int, target_layer: int) -> Prediction:
        raise NotImplementedError


class SyntheticPredictor(Predictor):
    def __init__(
        self,
        model: V1ModelConfig,
        routing: RoutingProvider,
        accuracy: float,
        seed: int,
    ) -> None:
        self.model = model
        self.routing = routing
        self.accuracy = accuracy
        self.seed = seed

    def predict(self, token_id: int, source_layer: int, target_layer: int) -> Prediction:
        true_experts = tuple(self.routing.get_active_experts(token_id, target_layer))
        rng = random.Random(
            stable_seed(self.seed, 2, token_id, source_layer, target_layer)
        )
        retained: List[int] = [
            expert for expert in true_experts if rng.random() < self.accuracy
        ]
        candidates = [
            expert
            for expert in range(self.model.num_routed_experts)
            if expert not in true_experts
        ]
        wrong = rng.sample(candidates, self.model.top_k - len(retained))
        predicted = tuple(retained + wrong)
        return Prediction(
            expert_ids=predicted,
            true_expert_ids=true_experts,
            correct_count=len(retained),
        )
