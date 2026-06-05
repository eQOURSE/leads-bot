"""Tests for the ICP Strategist and ICP configs. No external calls."""

from __future__ import annotations

import pytest

from agents._models import IcpStrategy
from agents.icp_strategist import IcpStrategist


@pytest.fixture
def strategist(test_settings) -> IcpStrategist:
    return IcpStrategist(test_settings)


_SEGMENTS = ["tutrain", "eqourse_content", "eqourse_ai_data"]


def test_load_all_three_segments(strategist):
    loaded = [strategist.load_strategy(s) for s in _SEGMENTS]
    assert len(loaded) == 3
    assert all(isinstance(s, IcpStrategy) for s in loaded)
    assert {s.segment_name for s in loaded} == {
        "TUTRAIN",
        "eQOURSE Content",
        "eQOURSE AI Data Service",
    }


@pytest.mark.parametrize("segment", _SEGMENTS)
def test_scoring_weights_sum_to_100(strategist, segment):
    w = strategist.load_strategy(segment).scoring_weights
    total = w.funding_recency + w.segment_fit + w.buying_signal + w.reachability
    assert total == 100


@pytest.mark.parametrize("segment", _SEGMENTS)
def test_thresholds_in_order(strategist, segment):
    t = strategist.load_strategy(segment).scoring_thresholds
    assert t.auto_drop_below <= t.tier_2_above <= t.tier_1_above


@pytest.mark.parametrize("segment", _SEGMENTS)
def test_negative_signals_non_empty(strategist, segment):
    neg = strategist.load_strategy(segment).negative_signals
    assert len(neg) >= 3


@pytest.mark.parametrize("segment", _SEGMENTS)
def test_target_titles_no_duplicates(strategist, segment):
    titles = strategist.load_strategy(segment).target_titles
    lowered = [t.strip().lower() for t in titles]
    assert len(lowered) == len(set(lowered)), f"duplicate titles in {segment}"


def test_unknown_segment_raises(strategist):
    with pytest.raises(ValueError):
        strategist.load_strategy("invalid")


@pytest.mark.parametrize("segment", _SEGMENTS)
def test_pydantic_model_validates_real_config(strategist, segment):
    # Loading already validates; assert the round-trip dump is structurally sound.
    strategy = strategist.load_strategy(segment)
    dumped = strategy.model_dump(mode="json")
    revalidated = IcpStrategy.model_validate(dumped)
    assert revalidated.segment_name == strategy.segment_name


def test_segments_are_differentiated(strategist):
    """AI Data must target different industries/titles than TUTRAIN/Content."""
    tutrain = strategist.load_strategy("tutrain")
    ai_data = strategist.load_strategy("eqourse_ai_data")

    # Different NAICS codes
    assert set(tutrain.target_industries.naics_codes).isdisjoint(
        set(ai_data.target_industries.naics_codes)
    )
    # Different departments (AI data targets engineering/r&d/data)
    assert "engineering" in ai_data.target_departments
    assert "engineering" not in tutrain.target_departments


def test_list_segments(strategist):
    assert strategist.list_segments() == _SEGMENTS


@pytest.mark.asyncio
async def test_suggest_refinements_stub(strategist):
    result = await strategist.suggest_refinements("tutrain", [{"x": 1}, {"y": 2}])
    assert result["status"] == "not_yet_implemented"
    assert result["data_collected"] == 2
