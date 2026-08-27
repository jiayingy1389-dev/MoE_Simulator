# MoE V0 Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic single-layer, batch-1, decode-only MoE performance simulator with an explainable serial timeline, KV spill accounting, expert tiling, JSON output, and a tested synthetic example.

**Architecture:** Configuration, formulas, KV state, and timeline construction are separate modules. A single global event clock serializes DMA and compute; the orchestrator emits events in the exact V0 token and expert order. State-only lifecycle events have zero duration.

**Tech Stack:** Python 3.9, standard-library dataclasses/JSON/argparse, PyYAML 6, pytest 7.

## Global Constraints

- Implement one MoE layer, batch size 1, decode only.
- Prompt KV is initialized without prefill timing.
- Use one DMA/memory resource and one unified compute resource with no overlap.
- Execute Top-K experts serially in routing-trace order.
- Keep expert weights off chip; do not cache or prefetch them.
- Keep placed KV fixed during a request and release all KV after the request completes.
- Do not implement concurrency, migration, replacement, pipelines, predictors, a NoC, or other V1 behavior.
- Support Python 3.9; do not use newer typing syntax.

---

## File Map

- `simulator/__init__.py`: public package exports.
- `simulator/config.py`: typed configuration, YAML loading, and all pre-run validation.
- `simulator/model.py`: immutable event records and result summaries.
- `simulator/memory.py`: KV placement, traffic counters, peaks, and release.
- `simulator/compute.py`: byte, operation, tile, and cycle formulas.
- `simulator/timeline.py`: one-clock event builder and aggregate accounting.
- `simulator/simulator.py`: token/expert orchestration and summary production.
- `simulator/cli.py`: CLI, deterministic JSON serialization, and readable terminal summary.
- `configs/v0_synthetic.yaml`: small spill/repeated-expert/non-divisible-tile example.
- `tests/conftest.py`: reusable valid configuration factory.
- `tests/test_config.py`: validation and YAML tests.
- `tests/test_memory.py`: KV budget, placement, spill, peaks, and release tests.
- `tests/test_compute.py`: formula, tiling, and workspace tests.
- `tests/test_timeline.py`: duration and serialization tests.
- `tests/test_simulator.py`: execution order, traffic, determinism, repeated loads, and summaries.
- `tests/test_cli.py`: example CLI and JSON artifact test.
- `README.md`: model, assumptions, usage, output, limitations, and extension ideas.

---

### Task 1: Configuration and Result Types

**Files:**
- Create: `simulator/__init__.py`
- Create: `simulator/config.py`
- Create: `simulator/model.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelConfig`, `RequestConfig`, `HardwareConfig`, `SimulationConfig`, `ConfigError`, and `load_config(path: Path) -> SimulationConfig`.
- Produces: `TimelineEvent`, `TokenSummary`, `SequenceSummary`, and `SimulationResult`, each with `to_dict()`.

- [ ] **Step 1: Write failing configuration tests**

Create a fixture factory in `tests/conftest.py` that returns a nested mapping with `H=8`, `F=10`, `E=4`, `K=2`, `f_tile_size=4`, a two-token trace `[[0, 1], [1, 2]]`, and positive hardware values. In `tests/test_config.py`, assert:

```python
import pytest

from simulator.config import ConfigError, SimulationConfig


def test_rejects_capacity_partition(valid_config_dict):
    valid_config_dict["hardware"]["fixed_reserved_bytes"] = 900
    valid_config_dict["hardware"]["workspace_bytes"] = 200
    valid_config_dict["hardware"]["on_chip_capacity_bytes"] = 1024
    with pytest.raises(ConfigError, match="fixed_reserved_bytes.*workspace_bytes"):
        SimulationConfig.from_dict(valid_config_dict)


@pytest.mark.parametrize("field", ["off_chip_bytes_per_cycle", "compute_ops_per_cycle"])
def test_rejects_nonpositive_rates(valid_config_dict, field):
    valid_config_dict["hardware"][field] = 0
    with pytest.raises(ConfigError, match=field):
        SimulationConfig.from_dict(valid_config_dict)


def test_rejects_bad_routing_trace(valid_config_dict):
    valid_config_dict["request"]["routing_trace"] = [[0, 0], [1, 7]]
    with pytest.raises(ConfigError, match="routing_trace"):
        SimulationConfig.from_dict(valid_config_dict)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_config.py -v`

Expected: collection fails because `simulator.config` does not exist.

- [ ] **Step 3: Implement typed configuration and validation**

Use frozen dataclasses. `SimulationConfig.from_dict()` must construct nested objects and call `validate()`. Validation must check positive dimensions/capacities/rates/bit widths, non-negative startup and nonlinear cycles, `K <= E`, capacity partition, trace length, exactly K unique in-range IDs per trace row, and expert workspace via Task 2's formula. To avoid a circular dependency, place `required_workspace_bytes(model)` in `compute.py` in Task 2 and initially validate all constraints except workspace; Task 2 adds that single call.

The loader is exactly:

```python
def load_config(path: Path) -> SimulationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("cannot load config {}: {}".format(path, exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return SimulationConfig.from_dict(raw)
```

Define result dataclasses with only JSON-native fields after `to_dict()`. `TimelineEvent` fields are `event_id`, `depends_on`, `start_cycle`, `end_cycle`, `duration`, `token_id`, `stage`, `resource`, nullable `expert_id`, nullable `tile_id`, `bytes_transferred`, `operations`, `on_chip_kv_bytes`, and `off_chip_kv_bytes`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_config.py -v`

Expected: all configuration tests pass.

- [ ] **Step 5: Commit the task**

```text
git add simulator/__init__.py simulator/config.py simulator/model.py tests/conftest.py tests/test_config.py
git commit -m "feat: add MoE simulator configuration model"
```

---

### Task 2: Formula and KV Memory Models

**Files:**
- Create: `simulator/compute.py`
- Create: `simulator/memory.py`
- Create: `tests/test_compute.py`
- Create: `tests/test_memory.py`
- Modify: `simulator/config.py`

**Interfaces:**
- Consumes: config dataclasses from Task 1.
- Produces: `ceil_bytes(bits: int) -> int`, `f_tiles(F: int, tile_size: int) -> List[int]`, `dma_cycles(bytes_count, hardware) -> int`, `compute_cycles(operations, hardware) -> int`, `required_workspace_bytes(model) -> int`.
- Produces: `KVMemory.place_prompt(count)`, `place_decode_token(token_id) -> KVPlacement`, `off_chip_attention_bytes() -> int`, and `release()`.

- [ ] **Step 1: Write failing formula tests**

```python
from simulator.compute import f_tiles, expert_tile_metrics


def test_tail_tile_conserves_f():
    assert f_tiles(10, 4) == [4, 4, 2]
    assert sum(f_tiles(10, 4)) == 10


def test_expert_totals_match_theory(valid_config):
    metrics = [expert_tile_metrics(valid_config.model, size) for size in [4, 4, 2]]
    assert sum(item.gu_weight_bytes for item in metrics) == 2 * 8 * 10 * 8 // 8
    assert sum(item.down_weight_bytes for item in metrics) == 8 * 10 * 8 // 8
    assert sum(item.gu_operations for item in metrics) == 4 * 8 * 10
    assert sum(item.down_operations for item in metrics) == 2 * 8 * 10
```

- [ ] **Step 2: Write failing KV tests**

Construct `KVMemory` directly with a budget holding exactly two token KV records. Verify the first two placements are on chip, the third is off chip, current and peak values are correct, and `release()` zeros current values without changing peaks. Also assert:

```python
def test_spilled_kv_read_grows_with_context(memory_for_two_tokens):
    memory_for_two_tokens.place_prompt(2)
    third = memory_for_two_tokens.place_decode_token(0)
    assert third.location == "off_chip"
    assert memory_for_two_tokens.off_chip_attention_bytes() == third.bytes_count
    fourth = memory_for_two_tokens.place_decode_token(1)
    assert memory_for_two_tokens.off_chip_attention_bytes() == 2 * fourth.bytes_count
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m pytest tests/test_compute.py tests/test_memory.py -v`

Expected: imports fail because compute and memory modules do not exist.

- [ ] **Step 4: Implement formulas and memory state**

Use integer ceiling division `(numerator + denominator - 1) // denominator`. `expert_tile_metrics()` returns a frozen dataclass containing the four required metrics. `KVMemory` stores token locations in insertion order and maintains current/peak bytes plus cumulative off-chip write bytes. Prompt placement changes storage state but adds no timed write traffic; decoded off-chip placement increments off-chip write traffic.

The workspace formula is:

```python
def required_workspace_bytes(model: ModelConfig) -> int:
    partial = ceil_bytes(model.H * model.accumulator_bits)
    required = 0
    for tile_f in f_tiles(model.F, model.f_tile_size):
        metric = expert_tile_metrics(model, tile_f)
        gu_intermediate = ceil_bytes(2 * tile_f * model.activation_bits)
        nonlinear_output = ceil_bytes(tile_f * model.activation_bits)
        required = max(
            required,
            partial + metric.gu_weight_bytes + gu_intermediate,
            partial + metric.down_weight_bytes + nonlinear_output,
        )
    return required
```

Update `SimulationConfig.validate()` to raise `ConfigError` naming `workspace_bytes` and the required number when the reservation is too small.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_config.py tests/test_compute.py tests/test_memory.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the task**

```text
git add simulator/config.py simulator/compute.py simulator/memory.py tests/test_compute.py tests/test_memory.py
git commit -m "feat: model expert formulas and KV storage"
```

---

### Task 3: Serialized Timeline Builder

**Files:**
- Create: `simulator/timeline.py`
- Create: `tests/test_timeline.py`

**Interfaces:**
- Consumes: `TimelineEvent` and hardware timing formulas.
- Produces: `Timeline.add_dma(...)`, `add_compute(...)`, `add_nonlinear(...)`, and `add_state(...)`, all returning the appended event.

- [ ] **Step 1: Write failing timeline tests**

Create a timeline with DMA, compute, zero-cycle state, and nonlinear events. Assert event IDs are sequential, each dependency points to the previous event, every event has `end-start == duration`, the next event starts at the previous end, and totals satisfy `memory_cycles + compute_cycles == total_cycles`. Add an all-pairs interval assertion that positive-duration memory and compute events do not overlap:

```python
def overlaps(left, right):
    return left.start_cycle < right.end_cycle and right.start_cycle < left.end_cycle


for memory_event in memory_events:
    for compute_event in compute_events:
        assert not overlaps(memory_event, compute_event)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_timeline.py -v`

Expected: import fails because `simulator.timeline` does not exist.

- [ ] **Step 3: Implement the one-clock builder**

All public add methods call one private `_append(duration, ...)`. `_append` sets `start_cycle=self.current_cycle`, `end_cycle=start+duration`, `depends_on` to the preceding event ID or null, snapshots KV byte counts passed by the caller, appends the event, and advances `current_cycle`. DMA uses `dma_cycles`; ordinary compute uses `compute_cycles`; nonlinear uses the configured fixed cycles; state uses zero.

Maintain aggregate properties by summing events by resource/stage rather than a second mutable counter. This keeps the timeline as the accounting source of truth.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_timeline.py -v`

Expected: all timeline tests pass.

- [ ] **Step 5: Commit the task**

```text
git add simulator/timeline.py tests/test_timeline.py
git commit -m "feat: add serialized event timeline"
```

---

### Task 4: MoE Decode Orchestrator

**Files:**
- Create: `simulator/simulator.py`
- Create: `tests/test_simulator.py`

**Interfaces:**
- Consumes: `SimulationConfig`, `KVMemory`, formula helpers, and `Timeline`.
- Produces: `MoESimulator(config).run() -> SimulationResult`.

- [ ] **Step 1: Write failing execution-order and accounting tests**

For a one-token trace, assert the ordered stage subsequence is:

```python
[
    "qkv_attention_prepare",
    "place_token_kv",
    "attention_kv_read",
    "attention_compute",
    "router_compute",
    "expert_partial_sum_allocate",
    "expert_gu_weight_read",
    "expert_gu_compute",
    "expert_nonlinear",
    "expert_down_weight_read",
    "expert_down_compute",
    "expert_release",
    "topk_merge",
    "token_complete",
]
```

Allow repeated expert tile stages between the first and last expert lifecycle stages. Assert router operations equal `2*H*E`, Attention operations use the current context length, and each token latency equals its completion cycle minus its first event start.

- [ ] **Step 2: Write failing spill, repeated-load, release, and determinism tests**

Use a configuration that spills during decode. Assert later `attention_kv_read` event byte counts grow as off-chip KV grows. Run a two-token trace selecting the same expert and assert its GU plus Down weight-read bytes are charged once per selection. Run identical configs twice and assert `result.to_dict()` equality. Assert the final event is `release_request_kv`, current KV bytes are zero, and peaks remain non-zero.

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m pytest tests/test_simulator.py -v`

Expected: import fails because `simulator.simulator` does not exist.

- [ ] **Step 4: Implement token orchestration**

`run()` creates memory and timeline objects, initializes prompt placement without timed DMA, then loops through decoded tokens. For each token it emits exactly the design order. Omit zero-byte DMA events. Expert execution loops through the supplied expert IDs and all real F tile sizes, emitting allocation, GU read/compute, nonlinear, Down read/compute, and release events. Do not branch on expert identity except to label events.

After all tokens, call `memory.release()` and append `release_request_kv`. Derive summaries from the completed timeline and memory peaks. Define first spill position as the zero-based decoded `token_id` whose newly placed KV first goes off chip; prompt-only spill does not populate this decoded-token field.

- [ ] **Step 5: Run the full behavioral suite and verify GREEN**

Run: `python -m pytest tests/test_config.py tests/test_compute.py tests/test_memory.py tests/test_timeline.py tests/test_simulator.py -v`

Expected: all tests pass, including deterministic exact equality.

- [ ] **Step 6: Commit the task**

```text
git add simulator/simulator.py tests/test_simulator.py
git commit -m "feat: simulate serial MoE decode execution"
```

---

### Task 5: Synthetic Example, CLI, and Documentation

**Files:**
- Create: `simulator/cli.py`
- Create: `configs/v0_synthetic.yaml`
- Create: `tests/test_cli.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `load_config()` and `MoESimulator.run()`.
- Produces: `python -m simulator.cli CONFIG --output PATH`.

- [ ] **Step 1: Write the failing CLI test**

Invoke `main([config_path, "--output", output_path])` against the repository synthetic config. Assert return code zero, output JSON exists, `timeline` is non-empty, final stage is `release_request_kv`, terminal output contains `total_cycles`, and the example has a non-null decoded-token spill position.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_cli.py -v`

Expected: import fails because `simulator.cli` does not exist.

- [ ] **Step 3: Add the synthetic YAML**

Choose small integers satisfying workspace validation. Use `F=10` and `f_tile_size=4`, at least two decoded tokens, adjacent trace rows sharing one expert, and a KV budget that fits the prompt but spills during decode. Calculate the workspace reservation before fixing the final capacity values so the example is valid and keeps OCM values easy to inspect.

- [ ] **Step 4: Implement deterministic CLI output**

Use `argparse`. Write JSON with `indent=2`, `sort_keys=True`, and a trailing newline. Catch `ConfigError`, print `error: <message>` to stderr, and return 2. On success print total, memory, compute, nonlinear, KV traffic, expert-weight traffic, peaks, first spill, and per-token latency. Do not hide unexpected programming errors.

- [ ] **Step 5: Document V0**

README sections must cover purpose, installation requirements, synthetic command, test command, exact token and expert stage order, all configuration fields, event and summary fields, formulas, request-end KV release, zero-cost on-chip KV access, zero-duration state events, prompt initialization semantics, current limitations, and V1 extension suggestions that are explicitly not implemented.

- [ ] **Step 6: Run the full tests**

Run: `python -m pytest -v`

Expected: all tests pass with no warnings or errors.

- [ ] **Step 7: Run and inspect the example**

Run: `python -m simulator.cli configs/v0_synthetic.yaml --output outputs/v0_timeline.json`

Expected: exit code zero, a concise summary on stdout, and `outputs/v0_timeline.json` containing the complete timeline and summaries.

- [ ] **Step 8: Verify repository hygiene and formulas**

Run: `git diff --check`

Run: `python -m pytest -q`

Expected: no whitespace errors and all tests pass.

- [ ] **Step 9: Commit the task**

```text
git add simulator/cli.py configs/v0_synthetic.yaml tests/test_cli.py README.md outputs/v0_timeline.json
git commit -m "feat: add MoE V0 CLI and synthetic example"
```

---

## Final Verification

- [ ] Confirm every requested timeline field is present in JSON.
- [ ] Confirm all 11 user-required tests are represented and passing.
- [ ] Confirm request-end KV release is visible and current usage becomes zero.
- [ ] Confirm positive-duration memory and compute intervals never overlap.
- [ ] Confirm `memory_cycles + compute_cycles == total_cycles`.
- [ ] Confirm theoretical expert bytes and operations match timeline totals.
- [ ] Confirm repeated expert selection reloads weights.
- [ ] Confirm the README lists V0 omissions without implementing them.
- [ ] Request code review, resolve all critical and important findings, and rerun `python -m pytest -q`.
