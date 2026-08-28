import pytest

from simulator.v1.kv import KVCapacityExceeded, KVManager


def test_default_kv_grows_from_six_to_twenty_four_mib(v1_config):
    manager = KVManager(v1_config)
    assert manager.total_bytes == 6 * 1024 * 1024
    occupancies = [manager.total_bytes]
    capacities = [manager.cache_capacity_bytes]
    for _token in range(96):
        before = manager.total_bytes
        for layer_id in range(v1_config.model.num_layers):
            manager.add_layer_token(layer_id)
        assert manager.total_bytes - before == 192 * 1024
        occupancies.append(manager.total_bytes)
        capacities.append(manager.cache_capacity_bytes)
    assert manager.total_bytes == 24 * 1024 * 1024
    assert occupancies == sorted(occupancies)
    assert capacities == sorted(capacities, reverse=True)


def test_kv_is_tracked_per_layer(v1_config):
    manager = KVManager(v1_config)
    before = manager.layer_bytes(7)
    manager.add_layer_token(7)
    assert manager.layer_bytes(7) - before == 8 * 1024
    assert manager.layer_bytes(6) == before


def test_capacity_exceeded_has_required_error_code(v1_config_dict):
    v1_config_dict["hardware"]["on_chip_memory_mib"] = 9
    v1_config_dict["request"]["prompt_tokens"] = 0
    from simulator.v1.config import V1Config

    manager = KVManager(V1Config.from_dict(v1_config_dict))
    with pytest.raises(KVCapacityExceeded, match="KV_CAPACITY_EXCEEDED"):
        for _ in range(200):
            manager.add_layer_token(0)
