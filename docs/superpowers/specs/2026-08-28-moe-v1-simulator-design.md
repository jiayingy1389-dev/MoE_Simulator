# MoE V1 Cache and Prefetch Simulator Design

## 1. Goal and Compatibility

V1 extends the repository with a multi-layer, single-request decode simulator for dynamic on-chip KV growth, shrinking routed-Expert cache capacity, LRU eviction, demand loading, and one-layer-ahead speculative Expert prefetch.

V1 is additive. The existing V0 modules, configuration, CLI, outputs, report generator, and tests remain supported without behavior changes. V0 continues to run with:

```powershell
python -m simulator.cli configs/v0_synthetic.yaml --output outputs/v0_timeline.json
```

V1 uses an independent package and entry point:

```powershell
python -m simulator.v1.cli configs/v1_qwen_synthetic.yaml --output-dir outputs/v1
```

V1 is an architecture-level simulator. It does not model RTL, PE-internal dataflow, systolic arrays, AIE instructions, SCALE-Sim, real predictor networks, multiple requests, continuous batching, Expert tiling, multiple DMA engines, multiple compute units, or a detailed NoC.

## 2. Package Structure

```text
simulator/v1/
    __init__.py
    config.py
    model.py
    routing.py
    predictor.py
    expert.py
    cache.py
    kv.py
    scheduler.py
    simulator.py
    outputs.py
    plots.py
    cli.py
configs/
    v1_qwen_synthetic.yaml
    v1_qwen_baseline.yaml
tests/v1/
    conftest.py
    test_config.py
    test_expert.py
    test_routing.py
    test_predictor.py
    test_cache.py
    test_scheduler.py
    test_kv.py
    test_simulator.py
    test_outputs.py
```

The V1 package may reuse pure integer ceiling helpers from V0, but it does not reuse the V0 `Timeline`, whose single global clock intentionally prohibits overlap.

## 3. Model and Hardware Configuration

The default model is parameterized as:

```yaml
model_name: qwen1.5_moe_a2.7b_routed_only
num_layers: 24
total_parameters: 14.3e9
activated_parameters: 2.7e9
hidden_size: 2048
expert_intermediate_size: 1408
num_routed_experts: 60
top_k: 4
num_kv_heads: 16
head_dim: 128
weight_bits: 4
kv_bits: 16
batch_size: 1
num_requests: 1
attention_base_ops: 0
attention_ops_per_context_token: 8192
```

V1 models only the 60 routed experts. Shared experts and dense weights are represented by fixed/reserved resources and are not individually cached.

The default hardware is:

```yaml
clock_frequency_mhz: 300
on_chip_memory_mib: 32
fixed_reserved_mib: 3
expert_workspace_mib: 5
off_chip_bandwidth_gbps: 200
off_chip_dma_startup_cycles: 20
on_chip_read_bytes_per_cycle: 4096
compute_tops: 1.0
compute_ops_per_cycle: 3333
compute_startup_cycles: 1
```

`off_chip_bandwidth_gbps` is interpreted as decimal gigabytes per second despite the legacy field spelling. The derived rate is:

```text
off_chip_bytes_per_cycle
    = off_chip_bandwidth_gbps * 1,000,000,000
    / (clock_frequency_mhz * 1,000,000)
```

The default is approximately 666.67 bytes/cycle. `compute_tops` is descriptive metadata; execution uses the explicit integer `compute_ops_per_cycle` so cycle counts are deterministic.

Configuration validation requires batch size and request count to equal one, prefetch distance to equal one when enabled, positive dimensions/rates/capacities, `top_k <= num_routed_experts`, prediction accuracy in `[0, 1]`, sufficient workspace for one complete Expert, and prompt KV plus fixed/workspace reservations not to exceed total OCM.

## 4. Derived Quantities

All quantities are derived from configuration.

For one routed Expert:

```text
Gate/Up bytes = ceil(2 * H * F * weight_bits / 8)
Down bytes    = ceil(H * F * weight_bits / 8)
Expert bytes  = Gate/Up bytes + Down bytes

Gate/Up operations = 4 * H * F
Down operations    = 2 * H * F
```

The default yields 4,325,376 bytes and 17,301,504 operations.

Router operations are `2 * H * E`.

Attention uses the confirmed simplified model:

```text
attention_operations
    = attention_base_ops
    + context_length * attention_ops_per_context_token
```

The default `attention_ops_per_context_token` is `4 * hidden_size = 8192`. V1 does not separately model dense/QKV projection compute or dense-weight traffic.

One layer/token KV record is:

```text
kv_bytes_per_layer_token
    = ceil(2 * num_kv_heads * head_dim * kv_bits / 8)
```

The default is 8 KiB. Across 24 layers, each completed decode token adds 192 KiB.

## 5. Request and KV Lifecycle

The default request contains 32 prompt tokens and 96 decode tokens. Prefill execution is not simulated. At cycle zero, every layer is initialized with prompt KV, totaling 6 MiB.

For each decode token and each layer, V1 creates that layer's current-token 8 KiB KV before its Attention read. The current context length therefore includes the current token. After all 24 layers finish, total KV has increased by exactly 192 KiB. After 96 decode tokens, context length is 128 and total KV is exactly 24 MiB.

KV is permanently on chip for the request, has priority over Expert cache, is never evicted or offloaded, and is monotonically non-decreasing. Before each layer KV allocation, the cache evicts LRU Experts as required. The dynamic Expert-cache capacity is:

```text
total OCM - fixed reserved - Expert workspace - current KV
```

If capacity cannot be made available because total KV plus fixed/workspace reservations exceeds OCM, simulation records `kv_capacity_exceeded=true`, preserves partial outputs, and reports `KV_CAPACITY_EXCEEDED`.

## 6. Expert Segments and Compute

V1 defines:

```python
get_expert_segments(layer_id, expert_id) -> list[WeightSegment]
```

It returns exactly one segment representing the complete Gate, Up, and Down weights. A miss must finish transferring the full 4.125 MiB default Expert before that Expert begins compute. V1 does not stream tiles or overlap one Expert's internal DMA with its own compute.

After the full segment is available, Gate/Up and Down are separate serial Compute events. With default values, one Expert DMA lasts 6,509 cycles at 200 GB/s and 300 MHz. Gate/Up and Down together take approximately 5,194 cycles because each event pays compute startup.

## 7. Routing and Prediction

The replaceable routing interface is:

```python
RoutingProvider.get_active_experts(token_id, layer_id) -> list[int]
```

The default synthetic provider chooses four unique Experts from 60 for every token/layer using a fixed seed. Results are independent of call order: the provider derives a stable local seed from the configured routing seed, token ID, and layer ID. A future routing-source field is retained for CSV traces, but V1 implements only `synthetic`.

The replaceable predictor interface is:

```python
Predictor.predict(token_id, source_layer, target_layer) -> list[int]
```

For every Layer `l < num_layers - 1`, the predictor builds a unique Top-K set for Layer `l+1`. Each true Expert is independently retained with the configured probability; remaining positions are filled from non-true Experts using a deterministic local seed. Observed accuracy is the total intersection count divided by total predictions. Layer 23 makes no prediction.

`prefetch_enabled=false` disables prediction and prefetch traffic. The baseline configuration uses prefetch disabled and configured accuracy 0.0. Accuracy 0.0 with prefetch enabled remains a valid all-wrong speculation experiment.

## 8. Expert Cache

Cache keys are `(layer_id, expert_id)`. Identically numbered Experts from different layers are distinct entries.

Entries use these states:

```text
QUEUED_PREFETCH  # request only; no cache capacity consumed
LOADING          # DMA started; complete Expert capacity reserved
RESIDENT         # load complete and available
IN_COMPUTE       # real use in progress
```

Expert cache occupancy counts `LOADING`, `RESIDENT`, and `IN_COMPUTE` entries. Queued prefetch requests do not consume cache capacity.

True Expert use updates a monotonic `last_used_time`. LRU eviction selects the evictable resident entry with the smallest `(last_used_time, admission_order)` pair. LOADING and IN_COMPUTE entries cannot be evicted. Eviction requires no write-back because off-chip memory retains the full Expert copy.

Admission first evicts LRU residents until the full Expert fits in the current dynamic cache capacity. If normal cache capacity is smaller than one Expert, a demand load may reserve the complete Expert workspace, compute, and discard it immediately afterward. Prefetch cannot use workspace. If prefetch cannot be admitted when it is ready to start, it is skipped.

If KV growth is temporarily blocked only by protected LOADING/IN_COMPUTE entries, allocation waits until the earliest such protection ends, then retries eviction. A true arithmetic capacity excess produces `KV_CAPACITY_EXCEEDED`.

## 9. Three-Resource Scheduler

V1 schedules half-open events on three independent resources:

```text
OFF_CHIP_DMA
ON_CHIP_KV_READ
COMPUTE
```

Each resource serializes its own events. Events on different resources may overlap.

Off-chip DMA has one non-preemptive server and two FIFO priority queues. When idle it selects demand before prefetch. A running prefetch cannot be preempted by a later demand. Queued demand requests for all true misses are issued at Router resolution. Queued next-layer prefetch requests are issued at the same Router completion point after demands have been classified.

When a prefetch reaches the head of the DMA queue, cache admission is checked and reserved. A failed speculative admission skips the request without traffic. A started transfer cannot be canceled.

At the target Router completion:

- resident correct prediction is a hit;
- correct in-flight prediction waits for remaining DMA;
- unpredicted true Expert issues demand DMA;
- wrong queued prefetch is canceled without bytes;
- wrong in-flight prefetch completes and its bytes are wasted;
- wrong resident prefetch remains until LRU eviction.

`useful_prefetch_bytes` counts prefetched Experts actually used before eviction. A wrong prediction or a correct prefetched Expert evicted before use counts as `wasted_prefetch_bytes` once, at transfer completion or eviction classification as appropriate.

## 10. Layer Execution

For each token, layers run sequentially. A layer executes:

1. Allocate the current layer's 8 KiB KV, evicting LRU Experts if needed.
2. Schedule `context_length * 8 KiB` on `ON_CHIP_KV_READ`.
3. Schedule Attention Compute after that KV read and Compute availability.
4. Schedule Router Compute.
5. Resolve the current layer's actual Top-K against prior predictions.
6. Generate Layer `l+1` predictions and enqueue prefetch requests when enabled.
7. Enqueue demand loads for true misses with higher queue priority.
8. Execute the four true Experts serially, each waiting for its full segment, then Gate/Up and Down Compute.
9. Mark used cached Experts resident and update LRU; discard workspace-loaded Experts.

Layer 0 has no prior prediction, so its four misses use demand loads unless a previous token left matching Layer-0 Experts resident.

Prefetch for the next layer may overlap current-layer Expert Compute. On-chip KV reads are independent of off-chip DMA. Attention waits for its own layer KV read. V1 does not pre-read the next layer's KV or pipeline multiple tokens.

## 11. Output Records

### Per-token CSV

`v1_token_state.csv` contains:

```text
token_id, context_length, start_cycle, finish_cycle,
kv_occupancy_bytes, expert_cache_capacity_bytes,
expert_cache_occupancy_bytes, resident_expert_count,
kv_eviction_count, total_eviction_count
```

### Resource timeline CSV

`v1_resource_timeline.csv` contains:

```text
start_cycle, end_cycle, resource, operation,
token_id, layer_id, expert_id, bytes,
prefetch_or_demand, prediction_correct
```

Resource events are sorted by `(start_cycle, resource, sequence_id)` for deterministic output.

### Summary JSON

`v1_summary.json` contains all required metrics: cycles and KV sizes, cache hits/misses/rate, evictions, configured and observed prediction accuracy, useful/wasted/demand bytes, total Expert DMA and KV-read bytes, Expert wait, DMA/Compute overlap, and capacity status.

Byte accounting satisfies:

```text
total_expert_dma_bytes
    = useful_prefetch_bytes
    + wasted_prefetch_bytes
    + demand_expert_bytes
```

## 12. Plots

Matplotlib generates:

```text
v1_kv_and_expert_cache_over_time.png
v1_bandwidth_over_time.png
```

The capacity plot samples each token completion and shows KV occupancy, dynamic Expert-cache capacity, and actual Expert-cache occupancy against finish cycle.

The bandwidth plot has separate subplots. Each resource event is shown over its actual interval with event-average utilization:

```text
bytes / duration / resource_peak_bytes_per_cycle
```

Off-chip Expert DMA and on-chip KV read are never combined into one utilization value.

## 13. Testing and Acceptance

Tests cover all 17 required invariants plus full-segment behavior, bandwidth conversion, the 6,509-cycle Expert DMA, queued and in-flight wrong-prefetch handling, workspace fallback, per-resource non-overlap, deterministic output files, and readable PNG generation.

The complete existing V0 suite must remain green. Acceptance requires all tests to pass and the default 32-prompt/96-decode/24-layer experiment to generate all five output files deterministically.
