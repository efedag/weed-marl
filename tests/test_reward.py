
"""Unit tests for the weed-control reward function."""

import pytest

from marl.reward import RewardConfig, calculate_reward


def test_successful_spray():
    assert calculate_reward(weeds_sprayed=1) == pytest.approx(9.9)


def test_unnecessary_spray():
    assert calculate_reward(unnecessary_sprays=1) == pytest.approx(-5.1)


def test_step_penalty():
    assert calculate_reward() == pytest.approx(-0.1)


def test_multiple_weeds():
    assert calculate_reward(weeds_sprayed=2) == pytest.approx(19.9)


def test_combined_events():
    assert calculate_reward(
        weeds_sprayed=2,
        unnecessary_sprays=1,
    ) == pytest.approx(14.9)


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        calculate_reward(weeds_sprayed=-1)


def test_custom_reward_config():
    config = RewardConfig(
        weed_sprayed_reward=5.0,
        unnecessary_spray_penalty=-2.0,
        step_penalty=-0.5,
    )

    assert calculate_reward(
        weeds_sprayed=1,
        config=config,
    ) == pytest.approx(4.5)
