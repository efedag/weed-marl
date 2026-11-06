"""PettingZoo ParallelEnv arayüzünü gösteren küçük örnek ortam.

8x8 ızgarada bir drone ve bir yer robotu hareket eder. Gözlem, ajanın
(satır, sütun) konumudur. Bölüm, sabit adım sınırına ulaşınca biter.

Çalıştırma: python -m sim.example_parallel_env
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GRID_SIZE = 8
MAX_STEPS = 50

# Eylem -> (satır değişimi, sütun değişimi)
STAY, UP, DOWN, LEFT, RIGHT = range(5)
MOVES = {
    STAY: (0, 0),
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class ExampleParallelEnv(ParallelEnv):
    """1 drone + 1 yer robotu içeren örnek ızgara ortamı."""

    metadata = {"name": "example_parallel_env_v0", "render_modes": []}

    def __init__(self, grid_size=GRID_SIZE, max_steps=MAX_STEPS):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.possible_agents = ["drone_0", "ground_robot_0"]
        self.agents = []
        self.render_mode = None

        self._observation_spaces = {
            agent: spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int64)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(len(MOVES)) for agent in self.possible_agents
        }
        self._positions = {}
        self._step_count = 0
        self._rng = np.random.default_rng()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._positions = {
            agent: self._rng.integers(0, self.grid_size, size=2, dtype=np.int64)
            for agent in self.agents
        }

        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        for agent, action in actions.items():
            delta = np.array(MOVES[int(action)], dtype=np.int64)
            self._positions[agent] = np.clip(
                self._positions[agent] + delta, 0, self.grid_size - 1
            )

        self._step_count += 1
        time_up = self._step_count >= self.max_steps

        observations = {agent: self._observe(agent) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: time_up for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if time_up:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _observe(self, agent):
        return self._positions[agent].copy()


def run_random_episode(env, seed=None):
    """Rastgele eylemlerle bir bölüm koşar ve adım sayısını döndürür."""
    env.reset(seed=seed)
    for agent in env.possible_agents:
        env.action_space(agent).seed(seed)

    steps = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        steps += 1
    return steps


if __name__ == "__main__":
    print(f"Adim sayisi: {run_random_episode(ExampleParallelEnv())}")
