
"""Single ground robot baseline without drone assistance."""

from baselines.fixed_route import FixedRoutePolicy
from sim.example_parallel_env import STAY


class SingleRobotPolicy:
    """Control only the ground robot using a fixed scanning route."""

    def __init__(self, grid_size=8):
        self.robot_policy = FixedRoutePolicy(grid_size=grid_size)

    def reset(self):
        self.robot_policy.reset()

    def act(self, observations):
        """Return actions for both agents.

        The drone stays inactive while the ground robot scans.
        """
        return {
            "drone_0": STAY,
            "ground_robot_0": self.robot_policy.act(
                observations["ground_robot_0"]
            ),
        }
