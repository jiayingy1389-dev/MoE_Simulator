import pytest

from simulator.v1.predictor import SyntheticPredictor
from simulator.v1.routing import SyntheticRoutingProvider


def make_predictor(v1_config, accuracy, seed=1234):
    routing = SyntheticRoutingProvider(v1_config.model, v1_config.request.routing_seed)
    return SyntheticPredictor(v1_config.model, routing, accuracy, seed)


def test_accuracy_one_predicts_exact_true_set(v1_config):
    predictor = make_predictor(v1_config, 1.0)
    prediction = predictor.predict(0, 0, 1)
    assert prediction.correct_count == v1_config.model.top_k
    assert set(prediction.expert_ids) == set(prediction.true_expert_ids)


def test_accuracy_zero_has_no_true_experts(v1_config):
    predictor = make_predictor(v1_config, 0.0)
    prediction = predictor.predict(0, 0, 1)
    assert prediction.correct_count == 0
    assert set(prediction.expert_ids).isdisjoint(prediction.true_expert_ids)


@pytest.mark.parametrize("accuracy", [0.6, 0.8, 0.95])
def test_observed_accuracy_tracks_probability(v1_config, accuracy):
    predictor = make_predictor(v1_config, accuracy)
    predictions = [
        predictor.predict(token, layer, layer + 1)
        for token in range(40)
        for layer in range(8)
    ]
    observed = sum(item.correct_count for item in predictions) / (
        len(predictions) * v1_config.model.top_k
    )
    assert observed == pytest.approx(accuracy, abs=0.04)
    assert all(len(set(item.expert_ids)) == v1_config.model.top_k for item in predictions)


def test_prediction_is_call_order_independent(v1_config):
    first = make_predictor(v1_config, 0.8, 99)
    second = make_predictor(v1_config, 0.8, 99)
    keys = [(token, layer, layer + 1) for token in range(5) for layer in range(4)]
    expected = {key: first.predict(*key) for key in keys}
    actual = {key: second.predict(*key) for key in reversed(keys)}
    assert actual == expected
