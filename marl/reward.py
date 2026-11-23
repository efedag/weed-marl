
"""Reward function for multi-agent weed control."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    weed_sprayed_reward: float = 10.0
    unnecessary_spray_penalty: float = -5.0
    step_penalty: float = -0.1


def calculate_reward(
    weeds_sprayed: int = 0,
    unnecessary_sprays: int = 0,
    config: RewardConfig | None = None,
) -> float:
    """Calculate the reward for one simulation step.

    weeds_sprayed: Number of weeds successfully sprayed.
    unnecessary_sprays: Number of sprays without a valid weed.
    """
    if weeds_sprayed < 0 or unnecessary_sprays < 0:
        raise ValueError("Event counts cannot be negative.")

    if config is None:
        config = RewardConfig()

    return (
        weeds_sprayed * config.weed_sprayed_reward
        + unnecessary_sprays * config.unnecessary_spray_penalty
        + config.step_penalty
    )
