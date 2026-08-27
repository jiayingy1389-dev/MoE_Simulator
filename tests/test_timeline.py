from simulator.timeline import Timeline


def build_timeline(valid_config):
    timeline = Timeline(valid_config.hardware)
    timeline.add_dma(
        token_id=0,
        stage="kv_read",
        bytes_transferred=17,
        on_chip_kv_bytes=16,
        off_chip_kv_bytes=16,
    )
    timeline.add_state(
        token_id=0,
        stage="placement",
        on_chip_kv_bytes=16,
        off_chip_kv_bytes=16,
    )
    timeline.add_compute(
        token_id=0,
        stage="attention",
        operations=33,
        on_chip_kv_bytes=16,
        off_chip_kv_bytes=16,
    )
    timeline.add_nonlinear(
        token_id=0,
        stage="expert_nonlinear",
        expert_id=1,
        tile_id=0,
        on_chip_kv_bytes=16,
        off_chip_kv_bytes=16,
    )
    return timeline


def test_events_form_one_explicit_dependency_chain(valid_config):
    events = build_timeline(valid_config).events
    assert [event.event_id for event in events] == [0, 1, 2, 3]
    assert [event.depends_on for event in events] == [None, 0, 1, 2]


def test_duration_and_global_clock_are_conserved(valid_config):
    timeline = build_timeline(valid_config)
    for previous, event in zip(timeline.events, timeline.events[1:]):
        assert event.start_cycle == previous.end_cycle
    for event in timeline.events:
        assert event.end_cycle - event.start_cycle == event.duration
    assert timeline.events[0].duration == 4
    assert timeline.events[1].duration == 0
    assert timeline.events[2].duration == 3
    assert timeline.events[3].duration == 3
    assert timeline.total_cycles == 10


def test_memory_and_compute_intervals_never_overlap(valid_config):
    timeline = build_timeline(valid_config)
    memory_events = [event for event in timeline.events if event.resource == "memory"]
    compute_events = [event for event in timeline.events if event.resource == "compute"]
    for memory_event in memory_events:
        for compute_event in compute_events:
            overlaps = (
                memory_event.start_cycle < compute_event.end_cycle
                and compute_event.start_cycle < memory_event.end_cycle
            )
            assert not overlaps
    assert timeline.memory_cycles + timeline.compute_cycles == timeline.total_cycles
    assert timeline.nonlinear_cycles == 3


def test_event_captures_labels_metrics_and_kv_snapshot(valid_config):
    event = build_timeline(valid_config).events[-1]
    assert event.resource == "compute"
    assert event.stage == "expert_nonlinear"
    assert event.expert_id == 1
    assert event.tile_id == 0
    assert event.operations == 0
    assert event.bytes_transferred == 0
    assert event.on_chip_kv_bytes == 16
    assert event.off_chip_kv_bytes == 16
