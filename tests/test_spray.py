import numpy as np
import pytest

from marl.reward import calculate_reward
from sim.weed_env import CROP, DOWN, RIGHT, SOIL, SPRAY, STAY, WEED, WeedEnv, WeedEnvConfig

# 4x4 tarla: iki ot hücresi (0,0) ve (3,3), bir ürün hücresi (1,1)
FIELD = np.array(
    [
        [WEED, SOIL, SOIL, SOIL],
        [SOIL, CROP, SOIL, SOIL],
        [SOIL, SOIL, SOIL, SOIL],
        [SOIL, SOIL, SOIL, WEED],
    ],
    dtype=np.uint8,
)


@pytest.fixture
def make_env(tmp_path):
    path = tmp_path / "field.npy"
    np.save(path, FIELD)

    def factory(n_robots=2, **overrides):
        params = dict(grid_path=str(path), n_drones=1, n_robots=n_robots, max_steps=50)
        params.update(overrides)
        env = WeedEnv(WeedEnvConfig(**params))
        env.reset(seed=0)
        return env

    return factory


def assert_team_reward(rewards, expected):
    values = list(rewards.values())
    assert values == pytest.approx([expected] * len(values))


def place(env, *positions):
    env.positions[:] = positions


def test_drone_cannot_spray(make_env):
    env = make_env(n_robots=1)
    place(env, (0, 0), (2, 2))
    battery = env.battery.copy()
    _, rewards, _, _, _ = env.step({"drone_0": SPRAY, "robot_0": STAY})
    assert not env.sprayed.any()
    np.testing.assert_array_equal(env.battery, battery)
    assert env.weeds_sprayed == 0 and env.unnecessary_sprays == 0
    assert_team_reward(rewards, -0.1)


def test_spraying_a_weed_counts_as_weed_sprayed(make_env):
    env = make_env(n_robots=1)
    place(env, (2, 2), (0, 0))
    _, rewards, _, _, _ = env.step({"drone_0": STAY, "robot_0": SPRAY})
    assert env.sprayed[0, 0]
    assert (env.weeds_sprayed, env.unnecessary_sprays) == (1, 0)
    assert_team_reward(rewards, calculate_reward(1, 0))


@pytest.mark.parametrize("cell", [(1, 1), (2, 2)])  # ürün ve toprak
def test_spraying_a_non_weed_counts_as_unnecessary(make_env, cell):
    env = make_env(n_robots=1)
    place(env, (3, 0), cell)
    _, rewards, _, _, _ = env.step({"drone_0": STAY, "robot_0": SPRAY})
    assert (env.weeds_sprayed, env.unnecessary_sprays) == (0, 1)
    assert_team_reward(rewards, calculate_reward(0, 1))


def test_spraying_the_same_cell_twice_counts_once(make_env):
    env = make_env(n_robots=1)
    place(env, (2, 2), (0, 0))
    env.step({"drone_0": STAY, "robot_0": SPRAY})
    battery = env.battery[1]
    _, rewards, _, _, _ = env.step({"drone_0": STAY, "robot_0": SPRAY})
    assert (env.weeds_sprayed, env.unnecessary_sprays) == (1, 0)
    assert_team_reward(rewards, -0.1)
    # Tekrar ilaçlama sayılmaz ama batarya harcar
    assert env.battery[1] == battery - env.config.spray_cost


def test_two_robots_spraying_same_cell_in_same_step_count_once(make_env):
    env = make_env(n_robots=2)
    place(env, (2, 2), (3, 3), (3, 3))
    _, rewards, _, _, _ = env.step({"drone_0": STAY, "robot_0": SPRAY, "robot_1": SPRAY})
    assert (env.weeds_sprayed, env.unnecessary_sprays) == (1, 0)
    assert_team_reward(rewards, calculate_reward(1, 0))


def test_spray_costs_battery_and_is_masked_when_too_low(make_env):
    env = make_env(n_robots=1, robot_battery=10.0, spray_cost=3.0, robot_move_cost=1.0)
    place(env, (2, 2), (2, 2))
    env.step({"drone_0": STAY, "robot_0": SPRAY})
    assert env.battery[1] == 7.0
    env.battery[1] = 2.0
    mask = env.action_masks()[1]
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 1, 0])
    env.step({"drone_0": STAY, "robot_0": SPRAY})  # maskeli -> STAY
    assert env.unnecessary_sprays == 1 and env.battery[1] == 2.0


def test_termination_when_all_weeds_sprayed(make_env):
    env = make_env(n_robots=2)
    place(env, (2, 2), (0, 0), (3, 3))
    _, _, terminations, truncations, infos = env.step(
        {"drone_0": STAY, "robot_0": SPRAY, "robot_1": SPRAY}
    )
    assert all(terminations.values()) and not any(truncations.values())
    assert env.agents == []
    episode = infos["robot_0"]["episode"]
    assert episode["end_reasons"] == ["all_weeds_sprayed"]
    assert episode["completion_step"] == 1
    assert episode["weed_sprayed_ratio"] == 1.0
    assert episode["sprayed_area_ratio"] == pytest.approx(2 / 16)


def test_episode_summary_metrics(make_env):
    env = make_env(n_robots=1, max_steps=4)
    place(env, (2, 2), (0, 0))
    script = [SPRAY, RIGHT, SPRAY, DOWN]  # ot, sağa, toprak, aşağı
    for action in script:
        _, _, _, _, infos = env.step({"drone_0": RIGHT, "robot_0": action})
    episode = infos["robot_0"]["episode"]
    assert episode["weeds_total"] == 2
    assert episode["weeds_sprayed"] == 1
    assert episode["unnecessary_sprays"] == 1
    assert episode["weed_sprayed_ratio"] == 0.5
    assert episode["sprayed_area_ratio"] == pytest.approx(2 / 16)
    assert episode["path_length"] == {"drone_0": 1, "robot_0": 2}
    assert episode["path_length_total"] == 3
    assert episode["completion_step"] is None
    assert episode["truncated"] and not episode["terminated"]


def test_robot_stops_seeing_sprayed_weed_as_target(make_env):
    env = make_env(n_robots=1, detection_radius=3, obs_window=3, n_reports=2)
    place(env, (1, 1), (0, 0))
    obs, _, _, _, _ = env.step({"drone_0": STAY, "robot_0": STAY})
    assert obs["robot_0"][-6:].reshape(2, 3)[:, 2].tolist() == [1.0, 1.0]
    obs, _, _, _, _ = env.step({"drone_0": STAY, "robot_0": SPRAY})
    reports = obs["robot_0"][-6:].reshape(2, 3)
    # Kalan tek geçerli rapor (3,3) hücresi; (0,0) artık hedef değil
    assert reports[:, 2].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(reports[0, :2], [1.0, 1.0])
