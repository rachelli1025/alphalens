import pytest

from alphalens_core.pricing import cost_usd


def test_haiku_is_forty_times_cheaper_than_opus_for_same_tokens() -> None:
    haiku = cost_usd("claude-haiku-5-5", 10_000, 1_500)
    opus = cost_usd("claude-opus-5-5", 10_000, 1_500)
    assert haiku == pytest.approx(0.00175)
    assert opus == pytest.approx(0.07)
    assert opus is not None and haiku is not None
    assert opus / haiku == pytest.approx(40)


def test_unknown_model_has_no_cost() -> None:
    assert cost_usd("some-future-model", 1000, 1000) is None
