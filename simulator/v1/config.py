"""Typed configuration and validation for the V1 simulator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


MIB = 1024 * 1024


class V1ConfigError(ValueError):
    """Raised when a V1 configuration is invalid."""


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise V1ConfigError("{} must be a mapping".format(name))
    return value


def _value(raw: Mapping[str, Any], section: str, field: str) -> Any:
    if field not in raw:
        raise V1ConfigError("missing required field {}.{}".format(section, field))
    return raw[field]


def _int(raw: Mapping[str, Any], section: str, field: str) -> int:
    value = _value(raw, section, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise V1ConfigError("{}.{} must be an integer".format(section, field))
    return value


def _number(raw: Mapping[str, Any], section: str, field: str) -> float:
    value = _value(raw, section, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V1ConfigError("{}.{} must be numeric".format(section, field))
    return float(value)


def _string(raw: Mapping[str, Any], section: str, field: str) -> str:
    value = _value(raw, section, field)
    if not isinstance(value, str):
        raise V1ConfigError("{}.{} must be a string".format(section, field))
    return value


@dataclass(frozen=True)
class V1ModelConfig:
    model_name: str
    num_layers: int
    total_parameters: float
    activated_parameters: float
    hidden_size: int
    expert_intermediate_size: int
    num_routed_experts: int
    top_k: int
    num_kv_heads: int
    head_dim: int
    weight_bits: int
    kv_bits: int
    batch_size: int
    num_requests: int
    attention_base_ops: int
    attention_ops_per_context_token: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V1ModelConfig":
        return cls(
            model_name=_string(raw, "model", "model_name"),
            num_layers=_int(raw, "model", "num_layers"),
            total_parameters=_number(raw, "model", "total_parameters"),
            activated_parameters=_number(raw, "model", "activated_parameters"),
            hidden_size=_int(raw, "model", "hidden_size"),
            expert_intermediate_size=_int(raw, "model", "expert_intermediate_size"),
            num_routed_experts=_int(raw, "model", "num_routed_experts"),
            top_k=_int(raw, "model", "top_k"),
            num_kv_heads=_int(raw, "model", "num_kv_heads"),
            head_dim=_int(raw, "model", "head_dim"),
            weight_bits=_int(raw, "model", "weight_bits"),
            kv_bits=_int(raw, "model", "kv_bits"),
            batch_size=_int(raw, "model", "batch_size"),
            num_requests=_int(raw, "model", "num_requests"),
            attention_base_ops=_int(raw, "model", "attention_base_ops"),
            attention_ops_per_context_token=_int(
                raw, "model", "attention_ops_per_context_token"
            ),
        )


@dataclass(frozen=True)
class V1RequestConfig:
    prompt_tokens: int
    decode_tokens: int
    routing_source: str
    routing_seed: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V1RequestConfig":
        return cls(
            prompt_tokens=_int(raw, "request", "prompt_tokens"),
            decode_tokens=_int(raw, "request", "decode_tokens"),
            routing_source=_string(raw, "request", "routing_source"),
            routing_seed=_int(raw, "request", "routing_seed"),
        )


@dataclass(frozen=True)
class V1HardwareConfig:
    clock_frequency_mhz: float
    on_chip_memory_mib: int
    fixed_reserved_mib: int
    expert_workspace_mib: int
    off_chip_bandwidth_gbps: float
    off_chip_dma_startup_cycles: int
    on_chip_read_bytes_per_cycle: int
    compute_tops: float
    compute_ops_per_cycle: int
    compute_startup_cycles: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V1HardwareConfig":
        return cls(
            clock_frequency_mhz=_number(raw, "hardware", "clock_frequency_mhz"),
            on_chip_memory_mib=_int(raw, "hardware", "on_chip_memory_mib"),
            fixed_reserved_mib=_int(raw, "hardware", "fixed_reserved_mib"),
            expert_workspace_mib=_int(raw, "hardware", "expert_workspace_mib"),
            off_chip_bandwidth_gbps=_number(raw, "hardware", "off_chip_bandwidth_gbps"),
            off_chip_dma_startup_cycles=_int(
                raw, "hardware", "off_chip_dma_startup_cycles"
            ),
            on_chip_read_bytes_per_cycle=_int(
                raw, "hardware", "on_chip_read_bytes_per_cycle"
            ),
            compute_tops=_number(raw, "hardware", "compute_tops"),
            compute_ops_per_cycle=_int(raw, "hardware", "compute_ops_per_cycle"),
            compute_startup_cycles=_int(raw, "hardware", "compute_startup_cycles"),
        )

    @property
    def on_chip_memory_bytes(self) -> int:
        return self.on_chip_memory_mib * MIB

    @property
    def fixed_reserved_bytes(self) -> int:
        return self.fixed_reserved_mib * MIB

    @property
    def expert_workspace_bytes(self) -> int:
        return self.expert_workspace_mib * MIB

    @property
    def off_chip_bytes_per_cycle(self) -> float:
        return self.off_chip_bandwidth_gbps * 1_000_000_000 / (
            self.clock_frequency_mhz * 1_000_000
        )


@dataclass(frozen=True)
class V1PrefetchConfig:
    prefetch_enabled: bool
    prefetch_distance: int
    prediction_accuracy: float
    predictor_latency_cycles: int
    prediction_seed: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V1PrefetchConfig":
        enabled = _value(raw, "prefetch", "prefetch_enabled")
        if not isinstance(enabled, bool):
            raise V1ConfigError("prefetch.prefetch_enabled must be boolean")
        return cls(
            prefetch_enabled=enabled,
            prefetch_distance=_int(raw, "prefetch", "prefetch_distance"),
            prediction_accuracy=_number(raw, "prefetch", "prediction_accuracy"),
            predictor_latency_cycles=_int(raw, "prefetch", "predictor_latency_cycles"),
            prediction_seed=_int(raw, "prefetch", "prediction_seed"),
        )


@dataclass(frozen=True)
class V1Config:
    model: V1ModelConfig
    request: V1RequestConfig
    hardware: V1HardwareConfig
    prefetch: V1PrefetchConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V1Config":
        if not isinstance(raw, Mapping):
            raise V1ConfigError("configuration root must be a mapping")
        config = cls(
            V1ModelConfig.from_dict(_section(raw, "model")),
            V1RequestConfig.from_dict(_section(raw, "request")),
            V1HardwareConfig.from_dict(_section(raw, "hardware")),
            V1PrefetchConfig.from_dict(_section(raw, "prefetch")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_model = (
            "num_layers", "hidden_size", "expert_intermediate_size",
            "num_routed_experts", "top_k", "num_kv_heads", "head_dim",
            "weight_bits", "kv_bits", "attention_ops_per_context_token",
        )
        for field in positive_model:
            if getattr(self.model, field) <= 0:
                raise V1ConfigError("model.{} must be positive".format(field))
        if self.model.attention_base_ops < 0:
            raise V1ConfigError("model.attention_base_ops must be non-negative")
        if self.model.batch_size != 1:
            raise V1ConfigError("model.batch_size must equal 1")
        if self.model.num_requests != 1:
            raise V1ConfigError("model.num_requests must equal 1")
        if self.model.top_k > self.model.num_routed_experts:
            raise V1ConfigError("model.top_k must not exceed num_routed_experts")
        if self.request.prompt_tokens < 0 or self.request.decode_tokens <= 0:
            raise V1ConfigError("request token counts are invalid")
        if self.request.routing_source != "synthetic":
            raise V1ConfigError("request.routing_source currently supports only synthetic")

        hardware_positive = (
            "clock_frequency_mhz", "on_chip_memory_mib", "fixed_reserved_mib",
            "expert_workspace_mib", "off_chip_bandwidth_gbps",
            "on_chip_read_bytes_per_cycle", "compute_tops", "compute_ops_per_cycle",
        )
        for field in hardware_positive:
            if getattr(self.hardware, field) <= 0:
                raise V1ConfigError("hardware.{} must be positive".format(field))
        if self.hardware.off_chip_dma_startup_cycles < 0:
            raise V1ConfigError("hardware.off_chip_dma_startup_cycles must be non-negative")
        if self.hardware.compute_startup_cycles < 0:
            raise V1ConfigError("hardware.compute_startup_cycles must be non-negative")
        if self.prefetch.prefetch_enabled and self.prefetch.prefetch_distance != 1:
            raise V1ConfigError("prefetch.prefetch_distance must equal 1")
        if not 0.0 <= self.prefetch.prediction_accuracy <= 1.0:
            raise V1ConfigError("prefetch.prediction_accuracy must be in [0, 1]")
        if self.prefetch.predictor_latency_cycles < 0:
            raise V1ConfigError("prefetch.predictor_latency_cycles must be non-negative")

        from .expert import expert_bytes, kv_bytes_per_layer_token

        if self.hardware.expert_workspace_bytes < expert_bytes(self.model):
            raise V1ConfigError("hardware.expert_workspace is smaller than one Expert")
        initial_kv = (
            self.request.prompt_tokens
            * self.model.num_layers
            * kv_bytes_per_layer_token(self.model)
        )
        reserved = self.hardware.fixed_reserved_bytes + self.hardware.expert_workspace_bytes
        if initial_kv + reserved > self.hardware.on_chip_memory_bytes:
            raise V1ConfigError("initial prompt KV plus reserved memory exceeds total OCM")


def load_v1_config(path: Path) -> V1Config:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V1ConfigError("cannot load V1 config {}: {}".format(path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise V1ConfigError("configuration root must be a mapping")
    return V1Config.from_dict(raw)
