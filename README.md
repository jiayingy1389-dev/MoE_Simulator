# MoE V0 Performance Simulator

这是一个用于理解单层 Mixture-of-Experts 在 decode 阶段执行顺序的性能/架构模拟器。它不进行矩阵数值计算，只记录 operations、传输 bytes、依赖关系、起止周期和 KV 状态。

## 运行

需要 Python 3.9、PyYAML 和 pytest。

```powershell
python -m simulator.cli configs/v0_synthetic.yaml --output outputs/v0_timeline.json
```

测试命令：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

禁用 pytest 外部插件自动加载，是为了避免本机无关插件影响这个独立项目。

## V0 执行顺序

V0 只有一个全局时钟。任意正时长区间内只会有 Memory 或 Compute 之一活动；零周期 `state` 事件只描述状态变化。

每个 decode token 严格执行：

1. `qkv_attention_prepare`
2. `place_token_kv`
3. 若新 KV 位于片外，执行 `kv_write`
4. 若当前上下文存在片外 KV，执行 `attention_kv_read`
5. `attention_compute`
6. `router_compute`
7. 按 `routing_trace` 顺序串行执行 K 个 Expert
8. `topk_merge`
9. `token_complete`

每个 Expert 创建 Down partial sum 后，沿 F 维逐 tile 执行：

```text
GU weight DMA
→ GU compute
→ Nonlinear
→ Down weight DMA
→ Down compute 并累加到 partial sum
```

所有 tile 完成后释放 weight tile、中间数据和 partial sum。权重不缓存、不预取；同一 Expert 被连续 token 选择时仍会重新读取。

## KV 与 OCM

KV 片内预算为：

```text
on_chip_capacity_bytes - fixed_reserved_bytes - workspace_bytes
```

每 token KV 大小为：

```text
ceil(2 * num_kv_heads * head_dim * kv_bits / 8)
```

prompt KV 在模拟开始前直接初始化，不模拟 prefill 或初始化传输时间。完整 token 能放入 KV 预算时放片内，否则放片外；单个 token 不拆分。

request 执行期间，已经放置的 KV 不迁移、不替换。最后一个 decode token 完成后，`release_request_kv` 零周期事件释放全部片内和片外 KV。当前使用量归零，但 peak 和累计流量保留。

V0 假设片内 KV 访问时间为 0。Attention 只为当前上下文中的片外 KV 产生 DMA 流量。当前 token 若 spill，会先写入片外，再在同一 token 的 Attention 中读回。

## Expert 与时间公式

实际 tile 大小为 `F_i` 时：

```text
GU weight bytes   = ceil(2 * H * F_i * weight_bits / 8)
Down weight bytes = ceil(H * F_i * weight_bits / 8)
GU operations     = 4 * H * F_i
Down operations   = 2 * H * F_i
```

规定 `1 MAC = 2 operations`。末尾 tile 使用真实余数，所有 `F_i` 之和严格等于 F。

```text
partial_sum = ceil(H * accumulator_bits / 8)
GU live     = partial_sum + GU weight + 2 * F_i activations
Down live   = partial_sum + Down weight + F_i activation
workspace required = max(所有 tile 的 GU live 与 Down live)
```

```text
DMA cycles = dma_startup_cycles
           + ceil(bytes / off_chip_bytes_per_cycle)

Compute cycles = compute_startup_cycles
               + ceil(operations / compute_ops_per_cycle)

Router operations = 2 * H * E

Attention operations = attention_base_ops
                     + context_length * attention_ops_per_context_token
```

Nonlinear 时长直接取 `nonlinear_cycles_per_tile`，它属于 compute cycles，也单独累计为 nonlinear cycles。

## 配置与校验

- `model`：H、F、E、K、KV head 形状、各类 bit width、F tile 大小、Attention 简化参数、prepare 和 merge operations。
- `request`：prompt/decode 长度，以及每个 token 的有序 Top-K Expert ID。每行必须恰好有 K 个互不重复且位于 `[0, E)` 的 ID。
- `hardware`：OCM 总容量、固定预留、workspace、片外带宽、compute 吞吐和各种启动/非线性周期。

不合法容量、非正带宽或吞吐、非法 trace、缺少字段及不足 workspace 会在模拟前给出明确错误。

## JSON 输出

每个 timeline event 包含：

```text
event_id, depends_on, start_cycle, end_cycle, duration, shape,
token_id, stage, resource, expert_id, tile_id,
bytes_transferred, operations,
on_chip_kv_bytes, off_chip_kv_bytes
```

事件使用半开区间 `[start_cycle, end_cycle)`；`depends_on` 指向前一个事件，所以 V0 timeline 是一条显式依赖链。

sequence summary 包含 total/memory/compute/nonlinear cycles、片外 KV read/write bytes、Expert weight read bytes、片内/片外 KV peak，以及首次 decode spill token。每个 token 另有 latency、start 和 end。

## 时间轴表格

从现有 JSON 结果生成 CSV 和 Markdown 表格：

```powershell
python -m simulator.report outputs/v0_timeline.json `
  --csv outputs/v0_timeline_table.csv `
  --markdown outputs/v0_timeline_table.md
```

- `outputs/v0_timeline_table.csv` 适合用 Excel 筛选和绘图。
- `outputs/v0_timeline_table.md` 可以直接阅读完整的 85 个事件。

表格逐事件显示当前阶段、片上内存预留/占用和片外有效带宽。片上总量采用：

```text
fixed_reserved_bytes + workspace_reserved_bytes + on_chip_kv_bytes
```

这里的 Workspace 是配置预留量，不是 Expert 临时 buffer 的动态实测占用。Memory 事件的有效带宽为 `bytes_transferred / duration`，因此 DMA startup 周期也包含在分母中；Compute 和 state 事件的片外带宽为 0。

`configs/v0_synthetic.yaml` 使用 `F=10`、`f_tile_size=4`，因此 tile 为 `[4, 4, 2]`；两个 token 都选择 Expert 1；KV 预算只容纳两个 prompt token，所以第一个 decode token 开始 spill。

该配置的实际摘要为：

```text
total_cycles: 263
memory_cycles: 121
compute_cycles: 142
nonlinear_cycles: 36
off_chip_kv_read_bytes: 48
off_chip_kv_write_bytes: 32
expert_weight_read_bytes: 960
on_chip_kv_peak_bytes: 32
off_chip_kv_peak_bytes: 32
first_kv_spill_token: 0
token 0 latency: 131 cycles
token 1 latency: 132 cycles
```

生成的 `outputs/v0_timeline.json` 包含 85 个事件。开头可观察到 prepare、KV 放置、16-byte KV write、16-byte KV read、Attention 和 Router；最后一个事件是 cycle 263 上的零周期 `release_request_kv`，片内与片外 KV 当前使用量均为 0。

## V0 未模拟内容

- 真实 Router 和矩阵数值
- prefill 时间
- Expert 权重缓存、预取或替换
- KV 迁移或替换
- DMA/Compute 重叠、流水线
- 多 channel、多 compute unit、NoC
- batch 大于 1、并发 request
- 精确 Attention kernel、具体 VEK280 参数

## 后续建议（未实现）

V1 可在保留现有事件和公式接口的前提下，引入双资源重叠调度、Expert 权重缓存/预取、多 DMA channel 或多 compute unit。扩展前应先增加资源冲突与依赖测试，避免改变 V0 串行语义。
