# MoE V0 Performance Simulator Design

## 1. Goal and Scope

Build a small, deterministic performance/architecture simulator for one Mixture-of-Experts layer during autoregressive decode. The simulator records shapes, operation counts, transferred bytes, dependencies, and cycle ranges. It does not perform numerical matrix computation.

V0 supports one request at a time, batch size 1, one MoE layer, decode only, one unified compute resource, one DMA/off-chip-memory resource, and one unified on-chip memory (OCM). DMA and compute are strictly serialized. Top-K experts are executed serially in the order supplied by the routing trace.

V0 deliberately excludes predictor-based routing, prefetch, pipelining, overlapping resources, multiple memory channels, multiple compute units, a NoC model, concurrent requests, expert-weight caching, and expert-weight replacement policies.

## 2. Technology and Repository Layout

The implementation targets Python 3.9 and uses PyYAML for configuration and pytest for tests. It exports a JSON timeline and prints a concise sequence summary to the terminal.

```text
simulator/
    __init__.py
    config.py
    model.py
    memory.py
    compute.py
    timeline.py
    simulator.py
    cli.py
configs/
    v0_synthetic.yaml
tests/
    test_memory.py
    test_expert.py
    test_timeline.py
    test_simulator.py
README.md
```

Each module has one responsibility:

- `config.py` defines configuration objects, loads YAML, and validates constraints.
- `model.py` defines timeline events and summary result objects.
- `memory.py` owns KV placement, spill state, peaks, and request-lifetime release.
- `compute.py` owns tile construction and byte/operation/cycle formulas.
- `timeline.py` appends serialized events and advances the global clock.
- `simulator.py` implements the required decode and expert execution order.
- `cli.py` loads a configuration, runs the simulator, writes JSON, and prints the summary.

## 3. Configuration Model

### Model parameters

- `H`: hidden dimension
- `F`: expert intermediate dimension
- `E`: expert count
- `K`: Top-K count
- `num_kv_heads`
- `head_dim`
- `weight_bits`
- `kv_bits`
- `activation_bits`
- `accumulator_bits`
- `f_tile_size`
- `attention_base_ops`
- `attention_ops_per_context_token`
- `qkv_attention_prepare_ops`
- `topk_merge_ops`

### Request parameters

- `initial_prompt_length`
- `decode_length`
- `routing_trace`, with one ordered list of exactly `K` unique expert IDs per decoded token

### Hardware parameters

- `on_chip_capacity_bytes`
- `fixed_reserved_bytes`
- `workspace_bytes`
- `off_chip_bytes_per_cycle`
- `dma_startup_cycles`
- `compute_ops_per_cycle`
- `compute_startup_cycles`
- `nonlinear_cycles_per_tile`

All dimensions, capacities, bit widths, bandwidth, throughput, and tile sizes are positive integers. Startup-cycle values and nonlinear cycles are non-negative integers. `K` cannot exceed `E`. Every routing trace entry has exactly `K` unique IDs in `[0, E)`, and trace length equals `decode_length`.

The implementation rejects configurations where:

```text
fixed_reserved_bytes + workspace_bytes > on_chip_capacity_bytes
```

Errors name the invalid field or violated relationship.

## 4. Event-Based Serial Architecture

The simulator uses a direct event builder rather than a cycle-by-cycle engine or a general discrete-event scheduler. It holds a single global clock. Appending a DMA or compute event calculates its duration, records `[start_cycle, end_cycle)`, and advances the clock. Consequently, positive-duration memory and compute events cannot overlap by construction.

State changes such as allocating a buffer, placing KV, completing a token, or releasing request state are zero-duration events with resource `state`. They make lifetimes visible without contributing to memory or compute cycles.

Every event contains at least:

```text
start_cycle
end_cycle
duration
shape
token_id
stage
resource
expert_id
tile_id
bytes_transferred
operations
on_chip_kv_bytes
off_chip_kv_bytes
```

`expert_id` and `tile_id` are null when they do not apply. Dependencies are represented by the event's predecessor event ID; the strictly serial V0 timeline therefore forms one explicit chain.

## 5. KV Placement and Lifetime

The KV-only on-chip budget is:

```text
kv_budget = on_chip_capacity_bytes - fixed_reserved_bytes - workspace_bytes
```

The physical bytes for one token's K and V tensors are calculated by rounding their total bit count up to a whole byte:

```text
kv_bytes_per_token = ceil(2 * num_kv_heads * head_dim * kv_bits / 8)
```

Prompt tokens are placed at initialization, in order. Each decoded token is placed before its Attention computation. A complete token is placed on chip if the remaining KV budget can hold it; otherwise it is placed off chip. A placement is never split across memory levels.

During a request, already placed KV is neither migrated nor replaced. Once all `decode_length` tokens finish, a zero-cycle `release_request_kv` state event releases all on-chip and off-chip KV and reports both current usages as zero. Peak usage and accumulated traffic remain available in the summary after release.

Attention reads all off-chip KV belonging to the prompt and decoded-token context. On-chip KV access costs zero cycles and produces no off-chip traffic in V0; this simplifying assumption is documented in the README.

If a decoded token is placed off chip, its KV write DMA occurs before the Attention KV read. Because current-token KV belongs to the Attention context, the new off-chip KV contributes to that same token's read traffic.

## 6. Token Execution Flow

For zero-based decoded `token_id`, the Attention context length is:

```text
context_length = initial_prompt_length + token_id + 1
```

Each token generates events in this exact order:

1. QKV/Attention preparation compute.
2. Current-token KV placement state update.
3. Current-token KV write DMA when it was placed off chip.
4. DMA read of all off-chip KV in the current Attention context, omitted when the byte count is zero.
5. Attention compute.
6. Router compute.
7. Resolve the ordered Top-K expert IDs from `routing_trace[token_id]`.
8. Execute each selected expert serially.
9. Top-K output merge compute.
10. Token-complete state event.

The operation counts are:

```text
attention_ops = attention_base_ops
              + context_length * attention_ops_per_context_token
router_ops = 2 * H * E
```

QKV/Attention preparation and Top-K merge use the configured `qkv_attention_prepare_ops` and `topk_merge_ops` respectively.

## 7. Expert Tiling and Workspace

Each selected expert splits `F` into ordered tiles of at most `f_tile_size`. The final tile uses its actual remaining size `F_i`.

For every tile:

```text
GU weight bytes   = ceil(2 * H * F_i * weight_bits / 8)
Down weight bytes = ceil(H * F_i * weight_bits / 8)
GU operations     = 4 * H * F_i
Down operations   = 2 * H * F_i
```

One MAC equals two operations. Expert weights always reside off chip, are loaded only after routing resolves, are never prefetched or cached, and are loaded again even when consecutive tokens select the same expert.

An expert executes:

1. Allocate and clear its Down partial-sum buffer as a zero-cycle state event.
2. For each tile: GU weight DMA, GU compute, nonlinear compute, Down weight DMA, and accumulating Down compute.
3. Release the temporary weight tile, intermediate values, and partial-sum buffer as a zero-cycle state event.

The simulator does not calculate the numerical expert output. The partial-sum lifetime records that the final output conceptually satisfies `Y = sum_i(Y_i)`.

Workspace requirements are checked before simulation. For each real tile size:

```text
partial_sum_bytes     = ceil(H * accumulator_bits / 8)
gu_intermediate_bytes = ceil(2 * F_i * activation_bits / 8)
nonlinear_output_bytes = ceil(F_i * activation_bits / 8)

gu_live_bytes = partial_sum_bytes
              + gu_weight_bytes
              + gu_intermediate_bytes

down_live_bytes = partial_sum_bytes
                + down_weight_bytes
                + nonlinear_output_bytes

required_workspace_bytes = max(gu_live_bytes, down_live_bytes over all tiles)
```

The configuration is rejected when `workspace_bytes` is smaller than this requirement. The workspace reservation is separate from the KV budget, so total modeled OCM use cannot exceed `on_chip_capacity_bytes`.

## 8. Timing and Accounting

DMA duration is:

```text
dma_startup_cycles + ceil(bytes_transferred / off_chip_bytes_per_cycle)
```

Ordinary compute duration is:

```text
compute_startup_cycles + ceil(operations / compute_ops_per_cycle)
```

Nonlinear duration is exactly `nonlinear_cycles_per_tile`. Nonlinear events use the compute resource, are included in `compute_cycles`, and are also accumulated separately as `nonlinear_cycles`.

The simulator reports per-token latency and sequence totals for:

- total, memory, compute, and nonlinear cycles;
- off-chip KV read and write bytes;
- expert-weight read bytes;
- peak on-chip and off-chip KV bytes;
- first decoded token position whose newly generated KV spills off chip, or null if no decoded token spills.

Memory and compute cycle totals partition all positive-duration events, so their sum equals total cycles in V0. Zero-duration state events do not affect the total.

## 9. Output

The CLI accepts a YAML configuration and an output JSON path. The JSON contains configuration-derived metadata, the full event timeline, per-token summaries, and the sequence summary. Terminal output presents the same headline totals plus the first KV spill position.

The synthetic configuration is deliberately small, includes a non-divisible final F tile, selects a repeated expert across adjacent tokens, and causes KV to spill during decode. It therefore demonstrates the most important V0 behaviors in one run.

## 10. Error Handling and Tests

Validation occurs before events are generated, preventing partial simulation output for invalid configurations. Invalid capacity, bandwidth, throughput, dimensions, routing traces, and insufficient expert workspace raise clear configuration errors. The CLI reports them without a Python traceback and exits non-zero.

Automated tests cover:

1. KV remains on chip while it fits the KV budget.
2. New KV goes off chip after the budget cannot hold another complete token.
3. Modeled OCM use never exceeds capacity.
4. Total GU and Down weight bytes equal their formulas.
5. Total GU and Down operations equal their formulas.
6. A non-divisible `F` produces a conserving final tile.
7. Positive-duration memory and compute intervals never overlap.
8. A repeated expert selection reloads weights for every token.
9. Off-chip KV Attention reads grow with context after spill.
10. Identical configuration and routing trace yield identical results.
11. Invalid capacity, bandwidth, and compute-throughput parameters fail clearly.
12. Request completion releases current KV usage while retaining peak statistics.
13. Invalid routing traces fail clearly.
14. Every event satisfies `end_cycle - start_cycle == duration`.
15. Summary values equal sums derived from timeline events.

## 11. Acceptance Criteria

The synthetic example runs from the command line, writes a readable JSON timeline, and prints a concise summary. All automated tests pass. Formulas conserve bytes, operations, tile sizes, and cycles. The README explains execution order, configuration fields, output fields, assumptions, limitations, test commands, and example commands without claiming V1 behavior.
