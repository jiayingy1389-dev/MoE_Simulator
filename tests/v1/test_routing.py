from simulator.v1.routing import SyntheticRoutingProvider


def test_synthetic_routing_is_unique_in_range_and_layer_sensitive(v1_config):
    provider = SyntheticRoutingProvider(v1_config.model, v1_config.request.routing_seed)
    first = provider.get_active_experts(0, 0)
    second_layer = provider.get_active_experts(0, 1)
    assert len(first) == v1_config.model.top_k
    assert len(set(first)) == len(first)
    assert all(0 <= expert < v1_config.model.num_routed_experts for expert in first)
    assert first != second_layer


def test_routing_is_reproducible_and_call_order_independent(v1_config):
    forward = SyntheticRoutingProvider(v1_config.model, 77)
    reverse = SyntheticRoutingProvider(v1_config.model, 77)
    keys = [(token, layer) for token in range(5) for layer in range(4)]
    expected = {key: forward.get_active_experts(*key) for key in keys}
    actual = {key: reverse.get_active_experts(*key) for key in reversed(keys)}
    assert actual == expected
