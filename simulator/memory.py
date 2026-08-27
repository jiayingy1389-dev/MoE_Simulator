"""Request-scoped KV placement and traffic accounting."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class KVPlacement:
    kind: str
    token_id: Optional[int]
    location: str
    bytes_count: int


class KVMemory:
    def __init__(self, kv_budget_bytes: int, kv_bytes_per_token: int) -> None:
        if kv_budget_bytes < 0:
            raise ValueError("kv_budget_bytes must be non-negative")
        if kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        self.kv_budget_bytes = kv_budget_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.on_chip_bytes = 0
        self.off_chip_bytes = 0
        self.on_chip_peak_bytes = 0
        self.off_chip_peak_bytes = 0
        self.off_chip_write_bytes = 0
        self.placements: List[KVPlacement] = []

    def place_prompt_token(self) -> KVPlacement:
        return self._place(kind="prompt", token_id=None, count_write=False)

    def place_decode_token(self, token_id: int) -> KVPlacement:
        return self._place(kind="decode", token_id=token_id, count_write=True)

    def _place(self, kind: str, token_id: Optional[int], count_write: bool) -> KVPlacement:
        fits = self.on_chip_bytes + self.kv_bytes_per_token <= self.kv_budget_bytes
        location = "on_chip" if fits else "off_chip"
        placement = KVPlacement(kind, token_id, location, self.kv_bytes_per_token)
        self.placements.append(placement)
        if fits:
            self.on_chip_bytes += self.kv_bytes_per_token
            self.on_chip_peak_bytes = max(self.on_chip_peak_bytes, self.on_chip_bytes)
        else:
            self.off_chip_bytes += self.kv_bytes_per_token
            self.off_chip_peak_bytes = max(self.off_chip_peak_bytes, self.off_chip_bytes)
            if count_write:
                self.off_chip_write_bytes += self.kv_bytes_per_token
        return placement

    def off_chip_attention_bytes(self) -> int:
        return self.off_chip_bytes

    def release(self) -> None:
        self.on_chip_bytes = 0
        self.off_chip_bytes = 0
