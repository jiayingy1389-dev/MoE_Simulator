"""Configuration types and validation for the V0 simulator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when a simulator configuration is invalid."""


def _mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping".format(name))
    return value


def _integer(section: Mapping[str, Any], section_name: str, field: str) -> int:
    if field not in section:
        raise ConfigError("missing required field {}.{}".format(section_name, field))
    value = section[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("{}.{} must be an integer".format(section_name, field))
    return value


@dataclass(frozen=True)
class ModelConfig:
    H: int
    F: int
    E: int
    K: int
    num_kv_heads: int
    head_dim: int
    weight_bits: int
    kv_bits: int
    activation_bits: int
    accumulator_bits: int
    f_tile_size: int
    attention_base_ops: int
    attention_ops_per_context_token: int
    qkv_attention_prepare_ops: int
    topk_merge_ops: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        return cls(**{field: _integer(raw, "model", field) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RequestConfig:
    initial_prompt_length: int
    decode_length: int
    routing_trace: Tuple[Tuple[int, ...], ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RequestConfig":
        prompt = _integer(raw, "request", "initial_prompt_length")
        decode = _integer(raw, "request", "decode_length")
        if "routing_trace" not in raw:
            raise ConfigError("missing required field request.routing_trace")
        trace = raw["routing_trace"]
        if not isinstance(trace, (list, tuple)):
            raise ConfigError("request.routing_trace must be a sequence")
        rows = []
        for row in trace:
            if not isinstance(row, (list, tuple)):
                raise ConfigError("request.routing_trace rows must be sequences")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
                raise ConfigError("request.routing_trace expert IDs must be integers")
            rows.append(tuple(row))
        return cls(prompt, decode, tuple(rows))


@dataclass(frozen=True)
class HardwareConfig:
    on_chip_capacity_bytes: int
    fixed_reserved_bytes: int
    workspace_bytes: int
    off_chip_bytes_per_cycle: int
    dma_startup_cycles: int
    compute_ops_per_cycle: int
    compute_startup_cycles: int
    nonlinear_cycles_per_tile: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HardwareConfig":
        return cls(**{field: _integer(raw, "hardware", field) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    request: RequestConfig
    hardware: HardwareConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SimulationConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError("configuration root must be a mapping")
        config = cls(
            model=ModelConfig.from_dict(_mapping(raw, "model")),
            request=RequestConfig.from_dict(_mapping(raw, "request")),
            hardware=HardwareConfig.from_dict(_mapping(raw, "hardware")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_model = (
            "H",
            "F",
            "E",
            "K",
            "num_kv_heads",
            "head_dim",
            "weight_bits",
            "kv_bits",
            "activation_bits",
            "accumulator_bits",
            "f_tile_size",
            "attention_base_ops",
            "attention_ops_per_context_token",
            "qkv_attention_prepare_ops",
            "topk_merge_ops",
        )
        for field in positive_model:
            if getattr(self.model, field) <= 0:
                raise ConfigError("model.{} must be positive".format(field))

        if self.request.initial_prompt_length < 0:
            raise ConfigError("request.initial_prompt_length must be non-negative")
        if self.request.decode_length <= 0:
            raise ConfigError("request.decode_length must be positive")

        positive_hardware = (
            "on_chip_capacity_bytes",
            "off_chip_bytes_per_cycle",
            "compute_ops_per_cycle",
        )
        for field in positive_hardware:
            if getattr(self.hardware, field) <= 0:
                raise ConfigError("hardware.{} must be positive".format(field))
        nonnegative_hardware = (
            "fixed_reserved_bytes",
            "workspace_bytes",
            "dma_startup_cycles",
            "compute_startup_cycles",
            "nonlinear_cycles_per_tile",
        )
        for field in nonnegative_hardware:
            if getattr(self.hardware, field) < 0:
                raise ConfigError("hardware.{} must be non-negative".format(field))

        if self.model.K > self.model.E:
            raise ConfigError("model.K must not exceed model.E")
        if (
            self.hardware.fixed_reserved_bytes + self.hardware.workspace_bytes
            > self.hardware.on_chip_capacity_bytes
        ):
            raise ConfigError(
                "hardware.fixed_reserved_bytes + hardware.workspace_bytes "
                "must not exceed hardware.on_chip_capacity_bytes"
            )

        if len(self.request.routing_trace) != self.request.decode_length:
            raise ConfigError("request.routing_trace length must equal request.decode_length")
        for token_id, experts in enumerate(self.request.routing_trace):
            if len(experts) != self.model.K:
                raise ConfigError(
                    "request.routing_trace[{}] must contain exactly K expert IDs".format(token_id)
                )
            if len(set(experts)) != len(experts):
                raise ConfigError(
                    "request.routing_trace[{}] contains duplicate expert IDs".format(token_id)
                )
            if any(expert < 0 or expert >= self.model.E for expert in experts):
                raise ConfigError(
                    "request.routing_trace[{}] expert IDs must be in [0, E)".format(token_id)
                )


def load_config(path: Path) -> SimulationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("cannot load config {}: {}".format(path, exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return SimulationConfig.from_dict(raw)
