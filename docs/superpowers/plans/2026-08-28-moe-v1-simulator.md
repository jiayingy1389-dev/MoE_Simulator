# MoE V1 Cache and Prefetch Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic 24-layer, single-request V1 simulator for on-chip KV growth, shrinking LRU Expert cache, full-Expert demand/prefetch DMA, three-resource overlap, required CSV/JSON outputs, and two PNG plots without changing V0 behavior.

**Architecture:** V1 lives in an independent `simulator.v1` package. Pure configuration-derived formulas feed replaceable routing/predictor interfaces, a protected-entry LRU cache, a request-scoped KV manager, and a discrete-event scheduler with independent OFF_CHIP_DMA, ON_CHIP_KV_READ, and COMPUTE resources. Output and plotting modules consume immutable simulation records.

**Tech Stack:** Python 3.9, dataclasses, heap/queue utilities, PyYAML 6, pytest 7, Matplotlib 3.5.

## Global Constraints

- Preserve all V0 files, entry points, outputs, formulas, and tests.
- Model one request, batch size one, decode only, 24 sequential layers by default.
- Model only 60 routed Experts; shared Experts and dense weights are fixed/reserved resources.
- Keep KV on chip with highest priority; never evict or offload KV.
- Cache key is exactly `(layer_id, expert_id)`.
- Transfer one complete Expert segment before its compute; do not implement tile streaming.
- Use one non-preemptive off-chip DMA with demand priority over queued prefetch.
- Use one Compute and one independent on-chip KV-read resource.
- Use 200 GB/s as decimal gigabytes per second.
- Use `compute_ops_per_cycle` for deterministic timing; `compute_tops` is metadata.
- Keep all parameter-derived values out of scheduling constants.

---

## File Map

- `simulator/v1/config.py`: V1 dataclasses, YAML loader, unit conversion, validation.
- `simulator/v1/model.py`: keys, segments, resource events, token state, summary, result.
- `simulator/v1/expert.py`: Expert/KV/operation formulas and full-segment interface.
- `simulator/v1/routing.py`: replaceable routing interface and deterministic synthetic routing.
- `simulator/v1/predictor.py`: replaceable predictor and controlled deterministic predictor.
- `simulator/v1/kv.py`: per-layer KV accounting and capacity exception.
- `simulator/v1/cache.py`: protected-state LRU entries, admission, eviction, workspace fallback.
- `simulator/v1/scheduler.py`: three resource calendars and prioritized non-preemptive DMA.
- `simulator/v1/simulator.py`: token/layer orchestration and statistics.
- `simulator/v1/outputs.py`: deterministic token CSV, resource CSV, and summary JSON.
- `simulator/v1/plots.py`: required capacity and bandwidth PNGs.
- `simulator/v1/cli.py`: V1 command line.
- `configs/v1_qwen_synthetic.yaml`: accuracy-0.8 default experiment.
- `configs/v1_qwen_baseline.yaml`: prefetch-disabled baseline.
- `tests/v1/`: focused unit/integration tests mirroring module boundaries.

---

### Task 1: V1 Configuration, Records, and Derived Formulas

**Files:**
- Create: `simulator/v1/__init__.py`
- Create: `simulator/v1/config.py`
- Create: `simulator/v1/model.py`
- Create: `simulator/v1/expert.py`
- Create: `tests/v1/conftest.py`
- Create: `tests/v1/test_config.py`
- Create: `tests/v1/test_expert.py`

**Interfaces:**
- Produces: `V1Config`, `V1ModelConfig`, `V1RequestConfig`, `V1HardwareConfig`, `V1PrefetchConfig`, `V1ConfigError`, `load_v1_config(path)`.
- Produces: `ExpertKey`, `WeightSegment`, `ResourceEvent`, `TokenState`, `V1Summary`, `V1Result`.
- Produces: `expert_bytes`, `expert_operations`, `kv_bytes_per_layer_token`, `router_operations`, `attention_operations`, `dma_cycles`, `compute_cycles`, and `get_expert_segments`.

- [ ] **Step 1: Write failing configuration and formula tests**

Tests load a compact fixture plus the default YAML and assert:

```python
assert config.hardware.off_chip_bytes_per_cycle == pytest.approx(200e9 / 300e6)
assert expert_bytes(config.model) == 4_325_376
assert expert_operations(config.model) == 17_301_504
assert router_operations(config.model) == 245_760
assert kv_bytes_per_layer_token(config.model) == 8 * 1024
assert dma_cycles(4_325_376, config.hardware) == 6_509
assert len(get_expert_segments(3, 7, config.model)) == 1
assert get_expert_segments(3, 7, config.model)[0].bytes == 4_325_376
```

Reject batch/request counts other than one, workspace smaller than one Expert, invalid prefetch distance, prediction accuracy outside `[0,1]`, non-positive rates, and prompt KV plus fixed/workspace above OCM.

- [ ] **Step 2: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_config.py tests/v1/test_expert.py -q`

Expected: import failure because `simulator.v1` does not exist.

- [ ] **Step 3: Implement configuration and immutable records**

Use frozen dataclasses. Store MiB inputs as exact byte properties using `value * 1024 * 1024`. Store GB/s conversion as a float property. Resource names are constants `OFF_CHIP_DMA`, `ON_CHIP_KV_READ`, and `COMPUTE`.

`WeightSegment` contains `key`, `name="full_expert"`, `bytes`, and Gate/Up/Down operation counts. `get_expert_segments()` returns a one-element tuple/list and contains no tile loop.

- [ ] **Step 4: Run GREEN and all V0 tests**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_config.py tests/v1/test_expert.py tests -q`

Expected: new tests and all V0 tests pass.

- [ ] **Step 5: Commit**

```text
git add simulator/v1 tests/v1
git commit -m "feat: add V1 configuration and model formulas"
```

---

### Task 2: Deterministic Routing and Predictor

**Files:**
- Create: `simulator/v1/routing.py`
- Create: `simulator/v1/predictor.py`
- Create: `tests/v1/test_routing.py`
- Create: `tests/v1/test_predictor.py`

**Interfaces:**
- Produces abstract/protocol interfaces `RoutingProvider.get_active_experts(token_id, layer_id)` and `Predictor.predict(token_id, source_layer, target_layer)`.
- Produces `SyntheticRoutingProvider`, `SyntheticPredictor`, and `Prediction` with ordered unique Expert IDs plus correct-intersection count.

- [ ] **Step 1: Write failing routing tests**

Assert every set has exactly Top-K unique IDs in range, different layers use layer-sensitive results, calling in different orders returns identical results, and two equal seeds reproduce the complete 96×24 trace.

- [ ] **Step 2: Write failing predictor tests**

For accuracy 1.0, assert prediction equals the true set and no wrong Expert appears. For 0.0, assert intersection is empty. For 0.6/0.8/0.95, assert all sets remain unique/in-range and long-run observed accuracy is within a statistical tolerance of the configured probability. Repeat in reversed call order and assert identical predictions.

- [ ] **Step 3: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_routing.py tests/v1/test_predictor.py -q`

Expected: missing modules.

- [ ] **Step 4: Implement call-order-independent randomness**

Derive a stable integer seed with arithmetic mixing of base seed, token, source layer, target layer, and a stream tag; do not use Python's randomized `hash()`. Instantiate a local `random.Random` per request. Predictor iterates true Experts in stable order, performs one Bernoulli draw each, then samples the required number from sorted non-true candidates.

- [ ] **Step 5: Run GREEN and commit**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_routing.py tests/v1/test_predictor.py -q`

```text
git add simulator/v1/routing.py simulator/v1/predictor.py tests/v1/test_routing.py tests/v1/test_predictor.py
git commit -m "feat: add deterministic V1 routing and prediction"
```

---

### Task 3: KV Manager and Protected LRU Expert Cache

**Files:**
- Create: `simulator/v1/kv.py`
- Create: `simulator/v1/cache.py`
- Create: `tests/v1/test_kv.py`
- Create: `tests/v1/test_cache.py`

**Interfaces:**
- Produces `KVManager.initialize_prompt()`, `add_layer_token(layer_id)`, `total_bytes`, `layer_bytes`, and `cache_capacity_bytes`.
- Produces `ExpertCache.admit_loading`, `mark_resident`, `begin_compute`, `finish_compute`, `touch`, `evict_for_capacity`, `discard`, and `select_lru`.
- Produces `CacheEntryState`, `AdmissionResult`, `Eviction`, and `KVCapacityExceeded` whose message contains `KV_CAPACITY_EXCEEDED`.

- [ ] **Step 1: Write failing KV tests**

Assert initial default KV is 6 MiB, each layer adds 8 KiB, each completed 24-layer token adds 192 KiB, 96 tokens end at 24 MiB, occupancy is monotonic, cache capacity is monotonic decreasing, and a deliberately undersized config raises `KV_CAPACITY_EXCEEDED`.

- [ ] **Step 2: Write failing cache tests**

Use small integer Expert sizes to assert layer-aware keys, deterministic LRU order, touch-on-real-use, no write-back bytes, LOADING/IN_COMPUTE protection, eviction on capacity shrink, cache occupancy conservation, workspace fallback only for demand, and prefetch skip when a full Expert cannot fit.

- [ ] **Step 3: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_kv.py tests/v1/test_cache.py -q`

- [ ] **Step 4: Implement KV and cache state machines**

Represent capacity as a callable/current value passed into admission and shrink operations. Cache occupancy is the sum of complete Expert bytes for LOADING, RESIDENT, and IN_COMPUTE entries. Queue-only requests live in the scheduler, not the cache. Workspace has one full-Expert slot because default workspace is 5 MiB; reject overlapping workspace use.

- [ ] **Step 5: Run GREEN and commit**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_kv.py tests/v1/test_cache.py -q`

```text
git add simulator/v1/kv.py simulator/v1/cache.py tests/v1/test_kv.py tests/v1/test_cache.py
git commit -m "feat: add V1 KV and LRU Expert cache"
```

---

### Task 4: Three-Resource Scheduler and DMA Queue

**Files:**
- Create: `simulator/v1/scheduler.py`
- Create: `tests/v1/test_scheduler.py`

**Interfaces:**
- Produces `ResourceScheduler.schedule_compute`, `schedule_kv_read`, `enqueue_dma`, `cancel_queued_prefetch`, `advance_dma`, `drain_until`, and immutable `ResourceEvent` records.
- DMA request callbacks expose start/completion so cache reservation and admission can occur at actual start.

- [ ] **Step 1: Write failing resource tests**

Assert events on the same resource never overlap, independent resource events can overlap, Attention-style compute can depend on a KV-read end, and overlap duration is computed from interval intersections without double counting.

- [ ] **Step 2: Write failing priority/cancellation tests**

Queue prefetch then demand before the server starts and assert demand starts first. Start prefetch then enqueue demand and assert prefetch is not preempted. Cancel queued wrong prefetch and assert no event/bytes. Mark an in-flight prefetch wrong and assert it completes with full wasted bytes.

- [ ] **Step 3: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_scheduler.py -q`

- [ ] **Step 4: Implement event scheduling**

Maintain independent availability cycles for Compute and KV read. DMA maintains one running request plus demand/prefetch FIFO queues. `advance_dma(now)` completes all transfers ending by `now`, starts the next admissible request at `max(previous_end, enqueue_cycle)`, and invokes admission at start. Use monotonic sequence IDs for deterministic ties.

- [ ] **Step 5: Run GREEN and commit**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_scheduler.py -q`

```text
git add simulator/v1/scheduler.py tests/v1/test_scheduler.py
git commit -m "feat: add V1 three-resource scheduler"
```

---

### Task 5: V1 Token/Layer Simulator

**Files:**
- Create: `simulator/v1/simulator.py`
- Create: `tests/v1/test_simulator.py`

**Interfaces:**
- Produces `V1Simulator(config, routing_provider=None, predictor=None).run() -> V1Result`.

- [ ] **Step 1: Write failing small-config integration tests**

Use 2–3 layers, a small Expert count, and short request. Assert layer order, per-layer KV allocation, context-sized KV reads, Router/expert operations, Layer-0 demand behavior, correct prefetch hit/wait behavior, workspace discard, cache hit/miss counts, and total on-chip capacity at every mutation.

- [ ] **Step 2: Write failing required-invariant tests**

Cover demand priority, actual DMA/Compute overlap, accuracy 1.0, prefetch-disabled and 0.0 baselines, identical-seed exact equality, byte conservation, expert wait cycles, observed accuracy, and KV-triggered eviction counting.

- [ ] **Step 3: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_simulator.py -q`

- [ ] **Step 4: Implement orchestration**

At each Router completion, first resolve current predictions/cancellations and enqueue true demands, then generate/enqueue next-layer predictions. Execute true Experts serially in routing order. Before each Expert compute, advance DMA until its load is complete; measure wait from readiness check to load completion. Touch cached Experts only on real use. At token completion append the exact required state row.

Finalize all DMA required for classified started transfers, classify unused prefetched entries, compute union-based DMA/Compute overlap, and derive all summary values from records/counters. Catch only `KVCapacityExceeded` inside `run()` to return partial results with the flag set.

- [ ] **Step 5: Run V1 and V0 GREEN**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_simulator.py tests -q`

- [ ] **Step 6: Commit**

```text
git add simulator/v1/simulator.py tests/v1/test_simulator.py
git commit -m "feat: simulate V1 cache and adjacent-layer prefetch"
```

---

### Task 6: Outputs, Plots, CLI, Configurations, and README

**Files:**
- Create: `simulator/v1/outputs.py`
- Create: `simulator/v1/plots.py`
- Create: `simulator/v1/cli.py`
- Create: `configs/v1_qwen_synthetic.yaml`
- Create: `configs/v1_qwen_baseline.yaml`
- Create: `tests/v1/test_outputs.py`
- Modify: `README.md`

**Interfaces:**
- Produces `write_v1_outputs(result, output_dir)` and `write_v1_plots(result, config, output_dir)`.
- CLI accepts config plus `--output-dir` and prints key metrics.

- [ ] **Step 1: Write failing output tests**

Assert exact CSV headers, deterministic line/order output, JSON required fields, two non-empty PNGs with valid PNG signatures and dimensions, separated bandwidth subplots, clear config errors, and partial output on `KV_CAPACITY_EXCEEDED`.

- [ ] **Step 2: Run RED**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/v1/test_outputs.py -q`

- [ ] **Step 3: Implement deterministic writers and plots**

Use explicit CSV field order, sorted-key indented JSON, Matplotlib's non-interactive `Agg` backend, step plots for token cache state, and separate bandwidth axes based on each event's `bytes/duration/peak_rate`. Close figures after saving.

- [ ] **Step 4: Add default and baseline YAML**

Default uses all confirmed Qwen/hardware/request values, routing seed, prediction seed 1234, accuracy 0.8, prefetch enabled, distance one, and zero predictor latency. Baseline is identical except prefetch disabled and configured accuracy 0.0. Include YAML comments about routed-only modeling and 200 GB/s units.

- [ ] **Step 5: Update README**

Document V0 preservation, V1 commands, all assumptions/config fields, complete-Expert-before-compute behavior, cache states, resource semantics, output fields, plot meanings, 200 GB/s conversion, shared/dense exclusion, baseline use, and unimplemented tile streaming.

- [ ] **Step 6: Run all tests**

Run: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q`

Expected: all V0 and V1 tests pass without warnings.

- [ ] **Step 7: Run default and baseline experiments**

```powershell
python -m simulator.v1.cli configs/v1_qwen_synthetic.yaml --output-dir outputs/v1
python -m simulator.v1.cli configs/v1_qwen_baseline.yaml --output-dir outputs/v1_baseline
```

Inspect required files, summary conservation, PNG decoding, and deterministic rerun hashes.

- [ ] **Step 8: Final review and commit**

Run `git diff --check`, `python -m compileall -q simulator`, and the full test suite. Review every original V1 requirement against a test or output field.

```text
git add simulator/v1 configs/v1_qwen_synthetic.yaml configs/v1_qwen_baseline.yaml tests/v1 README.md outputs/v1 outputs/v1_baseline
git commit -m "feat: add V1 outputs plots and default experiments"
```
