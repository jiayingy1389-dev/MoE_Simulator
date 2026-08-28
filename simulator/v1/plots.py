"""Non-interactive V1 capacity and bandwidth plots."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import V1Config
from .model import OFF_CHIP_DMA, ON_CHIP_KV_READ, V1Result


def write_v1_plots(result: V1Result, config: V1Config,
                   output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capacity_path = output_dir / "v1_kv_and_expert_cache_over_time.png"
    bandwidth_path = output_dir / "v1_bandwidth_over_time.png"
    cycles = [state.finish_cycle for state in result.token_states]

    fig, axis = plt.subplots(figsize=(10, 5))
    series = (
        ("KV occupancy", [s.kv_occupancy_bytes / 2**20 for s in result.token_states]),
        ("Expert cache capacity", [s.expert_cache_capacity_bytes / 2**20 for s in result.token_states]),
        ("Expert cache occupancy", [s.expert_cache_occupancy_bytes / 2**20 for s in result.token_states]),
    )
    for label, values in series:
        axis.step(cycles, values, where="post", label=label)
    axis.set(xlabel="Cycle", ylabel="MiB", title="On-chip KV and Expert Cache")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(capacity_path, dpi=120,
                metadata={"Software": "MoE Simulator V1"})
    plt.close(fig)

    resources = (
        (OFF_CHIP_DMA, config.hardware.off_chip_bytes_per_cycle,
         "Off-chip Expert DMA"),
        (ON_CHIP_KV_READ, config.hardware.on_chip_read_bytes_per_cycle,
         "On-chip KV read"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, (resource, peak, title) in zip(axes, resources):
        for event in sorted(result.events, key=lambda item: item.sequence_id):
            if event.resource != resource or event.end_cycle <= event.start_cycle:
                continue
            utilization = event.bytes / (event.end_cycle - event.start_cycle) / peak
            axis.hlines(utilization, event.start_cycle, event.end_cycle, linewidth=2)
        axis.set(ylabel="Utilization", title=title)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Cycle")
    fig.tight_layout()
    fig.savefig(bandwidth_path, dpi=120,
                metadata={"Software": "MoE Simulator V1"})
    plt.close(fig)
    return {"capacity": capacity_path, "bandwidth": bandwidth_path}
