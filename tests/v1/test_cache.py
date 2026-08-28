from simulator.v1.cache import CacheEntryState, ExpertCache
from simulator.v1.model import ExpertKey


def key(layer, expert):
    return ExpertKey(layer, expert)


def test_cache_key_includes_layer_id():
    cache = ExpertCache(expert_bytes=10, workspace_bytes=12)
    assert cache.reserve_load(key(0, 3), 20, "demand", 0).status == "cache"
    cache.mark_resident(key(0, 3))
    assert cache.reserve_load(key(1, 3), 20, "demand", 1).status == "cache"
    assert set(cache.entries) == {key(0, 3), key(1, 3)}


def test_lru_evicts_least_recently_used_expert():
    cache = ExpertCache(expert_bytes=10, workspace_bytes=12)
    for now, item in enumerate((key(0, 0), key(0, 1))):
        cache.reserve_load(item, 20, "demand", now)
        cache.mark_resident(item)
        cache.touch(item, now + 10)
    cache.touch(key(0, 1), 99)
    result = cache.reserve_load(key(0, 2), 20, "demand", 100)
    assert [eviction.key for eviction in result.evictions] == [key(0, 0)]
    assert key(0, 1) in cache.entries


def test_loading_and_computing_entries_are_not_evictable():
    cache = ExpertCache(expert_bytes=10, workspace_bytes=12)
    cache.reserve_load(key(0, 0), 10, "demand", 0)
    assert cache.entries[key(0, 0)].state == CacheEntryState.LOADING
    assert cache.reserve_load(key(0, 1), 10, "prefetch", 1).status == "skipped"
    assert cache.reserve_load(key(0, 1), 10, "demand", 1).status == "blocked"

    cache.mark_resident(key(0, 0))
    cache.begin_compute(key(0, 0), 2)
    assert cache.entries[key(0, 0)].state == CacheEntryState.IN_COMPUTE
    assert cache.reserve_load(key(0, 1), 10, "prefetch", 3).status == "skipped"


def test_demand_uses_workspace_only_when_cache_cannot_hold_one_expert():
    cache = ExpertCache(expert_bytes=10, workspace_bytes=12)
    demand = cache.reserve_load(key(0, 0), 5, "demand", 0)
    assert demand.status == "workspace"
    assert cache.workspace_key == key(0, 0)
    cache.begin_compute(key(0, 0), 1)
    cache.finish_compute(key(0, 0))
    assert cache.workspace_key is None
    assert key(0, 0) not in cache.entries

    prefetch = cache.reserve_load(key(0, 1), 5, "prefetch", 2)
    assert prefetch.status == "skipped"


def test_capacity_shrink_evicts_residents_but_not_protected_entries():
    cache = ExpertCache(expert_bytes=10, workspace_bytes=12)
    for item in (key(0, 0), key(0, 1)):
        cache.reserve_load(item, 20, "demand", 0)
        cache.mark_resident(item)
    evictions, blocked = cache.evict_for_capacity(10)
    assert len(evictions) == 1
    assert not blocked
    remaining = next(iter(cache.entries))
    cache.begin_compute(remaining, 5)
    evictions, blocked = cache.evict_for_capacity(0)
    assert not evictions
    assert blocked
