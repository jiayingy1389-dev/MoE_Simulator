from simulator.v1.model import COMPUTE, OFF_CHIP_DMA, ON_CHIP_KV_READ, ExpertKey
from simulator.v1.scheduler import ResourceScheduler, overlap_cycles


def test_resources_serialize_independently_and_can_overlap(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    compute_a = scheduler.schedule_compute(0, 10, "A", 0, 0)
    compute_b = scheduler.schedule_compute(2, 5, "B", 0, 0)
    kv_read = scheduler.schedule_kv_read(0, 7, 100, 0, 0)
    assert (compute_a.start_cycle, compute_a.end_cycle) == (0, 10)
    assert (compute_b.start_cycle, compute_b.end_cycle) == (10, 15)
    assert (kv_read.start_cycle, kv_read.end_cycle) == (0, 7)
    assert compute_a.resource == COMPUTE
    assert kv_read.resource == ON_CHIP_KV_READ
    assert overlap_cycles([compute_a, compute_b], [kv_read]) == 7


def test_demand_has_priority_over_queued_prefetch(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    scheduler.enqueue_dma(0, 50, ExpertKey(1, 1), "prefetch", 0, 0)
    scheduler.enqueue_dma(0, 20, ExpertKey(0, 2), "demand", 0, 0)
    event = scheduler.start_next_dma(0)
    assert event.prefetch_or_demand == "demand"
    assert event.expert_id == 2


def test_running_prefetch_is_not_preempted_by_later_demand(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    scheduler.enqueue_dma(0, 50, ExpertKey(1, 1), "prefetch", 0, 0)
    prefetch = scheduler.start_next_dma(0)
    scheduler.enqueue_dma(1, 20, ExpertKey(0, 2), "demand", 0, 0)
    assert scheduler.active_dma.event == prefetch
    completed = scheduler.advance_dma(prefetch.end_cycle)
    assert completed[0].prefetch_or_demand == "prefetch"
    assert scheduler.active_dma.event.prefetch_or_demand == "demand"
    assert scheduler.active_dma.event.start_cycle == prefetch.end_cycle


def test_queued_wrong_prefetch_can_be_cancelled_without_bytes(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    request = scheduler.enqueue_dma(0, 50, ExpertKey(1, 3), "prefetch", 0, 0)
    assert scheduler.cancel_queued_prefetch(request.key)
    assert scheduler.start_next_dma(0) is None
    assert not scheduler.events


def test_inflight_wrong_prefetch_completes_full_transfer(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    request = scheduler.enqueue_dma(0, 50, ExpertKey(1, 3), "prefetch", 0, 0)
    event = scheduler.start_next_dma(0)
    assert not scheduler.cancel_queued_prefetch(request.key)
    scheduler.mark_inflight_prefetch_wrong(request.key)
    completed = scheduler.advance_dma(event.end_cycle)
    assert completed[0].bytes == 50
    assert completed[0].prediction_correct is False
    assert completed[0].resource == OFF_CHIP_DMA


def test_temporarily_blocked_demand_remains_queued(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    key = ExpertKey(0, 1)
    request = scheduler.enqueue_dma(
        0, 10, key, "demand", 0, 0, on_start=lambda request, now: False
    )
    assert scheduler.start_next_dma(0) is None
    assert scheduler.demand_queue == [request]


def test_confirmed_prefetch_can_be_promoted_without_changing_accounting(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    request = scheduler.enqueue_dma(0, 10, ExpertKey(1, 2), "prefetch", 0, 1)
    assert scheduler.promote_queued_prefetch(request.key)
    event = scheduler.start_next_dma(0)
    assert event.prefetch_or_demand == "prefetch"
    assert not scheduler.prefetch_queue


def test_required_loads_follow_routing_order_across_original_queue_kinds(v1_config):
    scheduler = ResourceScheduler(v1_config.hardware)
    first = ExpertKey(1, 5)
    second = ExpertKey(1, 2)
    scheduler.enqueue_dma(0, 10, second, "prefetch", 0, 1)
    scheduler.enqueue_dma(0, 10, first, "demand", 0, 1)
    scheduler.prioritize_required([first, second])
    event = scheduler.start_next_dma(0)
    assert event.expert_id == first.expert_id
    scheduler.advance_dma(event.end_cycle)
    assert scheduler.active_dma.event.expert_id == second.expert_id
    assert scheduler.active_dma.event.prefetch_or_demand == "prefetch"
