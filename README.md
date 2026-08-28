# MoE V0/V1 Performance Simulator

这是一个用于理解 Mixture-of-Experts 在 decode 阶段执行顺序的性能/架构模拟器：V0 展示单层串行流程，V1 展示 24 层 cache/prefetch 与多资源重叠。它不进行矩阵数值计算，只记录 operations、传输 bytes、依赖关系、起止周期和 KV 状态。

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

## V0 的后续扩展说明

下面的 V1 已在独立 package 中实现双资源流量、Expert 权重缓存和预取，同时没有改变 V0 串行语义。多 DMA channel、多 compute unit 与 tile streaming 仍未实现。

## V1：多层 KV、Expert Cache 与预取

V1 是独立新增的 24 层、单 request、batch=1 decode 模拟器；V0 的命令、文件和串行语义均未改变。运行默认预取实验和 demand-only 对照实验：

```powershell
python -m simulator.v1.cli configs/v1_qwen_synthetic.yaml --output-dir outputs/v1
python -m simulator.v1.cli configs/v1_qwen_baseline.yaml --output-dir outputs/v1_baseline
```

默认模型只模拟 Qwen1.5-MoE-A2.7B 的 60 个 routed Experts，24 层、Top-4、H=2048、F=1408。shared Expert、dense/QKV 权重及其流量未逐项模拟，由固定预留空间抽象表示。request 含 32 个 prompt token 和 96 个 decode token；prefill 时间不模拟。

默认硬件为 32 MiB OCM，其中固定预留 3 MiB、完整 Expert workspace 5 MiB。片外带宽 `200 GB/s` 使用十进制字节，在 300 MHz 下约为 666.67 bytes/cycle；片上 KV read 为 4096 bytes/cycle；Compute 为 3333 operations/cycle。`compute_tops` 仅为说明字段，周期计算使用整数 `compute_ops_per_cycle`。

每个 routed Expert 的完整权重为 4,325,376 bytes：

```text
Expert bytes = ceil(3 * H * F * weight_bits / 8)
Gate/Up operations = 4 * H * F
Down operations = 2 * H * F
```

V1 每次必须先完成整个 Expert 的 DMA，再串行执行 Gate/Up 与 Down；没有 weight tile streaming。默认一次完整 DMA 为 6,509 cycles，Gate/Up 与 Down 计算分别约 3,462 和 1,732 cycles。后续版本才考虑“读够一个 tile 就开始算”。

每层、每 token 的 KV 为 8 KiB。prompt KV 初始总量 6 MiB；每个完整 decode token 在 24 层累计新增 192 KiB；96 个 decode token 后为 24 MiB。KV 在 request 生命周期内始终驻留片上、优先级高于 Expert cache；V1 不做 KV 迁移、逐出或片外 spill。CSV/summary 在 request 释放前采样，所以 `final_kv_bytes` 为 24 MiB；生命周期结束后的整体释放是模型假设，不额外生成资源事件。

动态 Expert cache 容量为：

```text
32 MiB - 3 MiB fixed - 5 MiB workspace - 当前 KV
```

cache key 是 `(layer_id, expert_id)`，不同层的同编号 Expert 不共享。状态包括 `LOADING`、`RESIDENT`、`REQUIRED` 和 `IN_COMPUTE`；加载中、Router 已确认属于本层 Top-K、以及计算中的 Expert 都不可逐出。可逐出项采用确定性 LRU，无 write-back。正常 cache 因受保护项暂时无法 admission 或容量小于一个 Expert 时，当前正在等待的真实 demand 可串行使用完整 workspace；prefetch 不能使用 workspace，也不能提前占用 workspace。每个非末层 Router 都会继续生成下一层预测，无法 admission 的 speculative 请求在 DMA head 处跳过。

V1 使用三个互相独立、各自串行的资源：

- `OFF_CHIP_DMA`：一个不可抢占 DMA；已排队的必需 load 优先于 speculative prefetch。
- `ON_CHIP_KV_READ`：读取该层当前完整 context 的 KV。
- `COMPUTE`：Attention、Router、Gate/Up 和 Down 共用一个计算单元。

不同资源的半开区间事件可以重叠。预测器在 Layer L 的 Router 后预测 L+1；错误的 queued prefetch 无流量取消，已经开始的错误传输必须完成并计入 wasted bytes。真实使用前完成的 prefetch 才计入 useful bytes。

每个输出目录包含：

- `v1_token_state.csv`：每个 decode token 的起止周期、context length、片上 KV、动态 Expert cache 容量/占用、resident 数和 eviction 数。
- `v1_resource_timeline.csv`：三个资源的逐事件起止周期、操作、层/Expert、bytes、demand/prefetch 与预测正确性。
- `v1_summary.json`：周期、KV、hit/miss、预测准确率、useful/wasted/demand 字节、等待和资源重叠。
- `v1_kv_and_expert_cache_over_time.png`：KV 增长、动态 cache 容量和实际 cache 占用。
- `v1_bandwidth_over_time.png`：片外 Expert DMA 与片上 KV read 分开的利用率子图。

字节统计满足：

```text
total_expert_dma_bytes
  = useful_prefetch_bytes + wasted_prefetch_bytes + demand_expert_bytes
```

当前默认完整实验的结果是：预取版 87,585,190 cycles，demand-only baseline 89,411,592 cycles，预取使总周期下降约 2.04%（baseline/预取约 1.021×）。预取版实际使用 13,287,555,072 useful prefetch bytes，同时产生 2,119,434,240 wasted prefetch bytes；Expert cache hit rate 为 33.33%。这说明在 32 MiB OCM 和完整 Expert 加载限制下，跨层预取仍能隐藏一部分 demand 等待，但容量压力和错误预测会显著削弱收益；后续 tile streaming 可能进一步改善重叠。
