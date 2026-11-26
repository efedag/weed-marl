from pathlib import Path

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

import sim.weed_env as weed_env
from sim.weed_env import (
    DOWN,
    LEFT,
    N_ACTIONS,
    RIGHT,
    SPRAY,
    STAY,
    UP,
    WEED,
    WeedEnv,
    WeedEnvConfig,
    observation_size,
    state_size,
)

GRID = Path("data/grids_64/parcel_000_first000_gt_64x64.npy")


def small_config(**overrides):
    params = dict(grid_size=16, max_steps=30)
    params.update(overrides)
    return WeedEnvConfig(**params)


def random_actions(env, rng):
    masks = env.action_masks()
    return {
        a: int(rng.choice(np.flatnonzero(masks[env.possible_agents.index(a)])))
        for a in env.agents
    }


def run_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    trace = [np.concatenate([obs[a] for a in env.possible_agents])]
    rewards = []
    while env.agents:
        obs, rew, _, _, _ = env.step(random_actions(env, rng))
        if obs:
            trace.append(np.concatenate([obs[a] for a in env.possible_agents]))
        rewards.append(list(rew.values()))
    return np.stack(trace), np.array(rewards), env.positions.copy(), env.state()


def test_parallel_api():
    parallel_api_test(WeedEnv(small_config()), num_cycles=40)


def test_parallel_api_with_npy_field_and_four_agents_each():
    config = WeedEnvConfig(grid_path=str(GRID), n_drones=4, n_robots=4, max_steps=20)
    parallel_api_test(WeedEnv(config), num_cycles=25)


def test_agent_names_and_order():
    env = WeedEnv(small_config(n_drones=3, n_robots=2))
    assert env.possible_agents == ["drone_0", "drone_1", "drone_2", "robot_0", "robot_1"]
    env.reset(seed=0)
    assert env.agents == env.possible_agents


@pytest.mark.parametrize("n_drones, n_robots", [(0, 2), (5, 2), (2, 0), (2, 5)])
def test_agent_count_limits(n_drones, n_robots):
    with pytest.raises(ValueError):
        WeedEnv(small_config(n_drones=n_drones, n_robots=n_robots))


def test_same_seed_same_episode():
    a = run_episode(WeedEnv(small_config()), seed=7)
    b = run_episode(WeedEnv(small_config()), seed=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_different_seed_different_field():
    env = WeedEnv(small_config())
    env.reset(seed=1)
    field_1 = env.field.copy()
    env.reset(seed=2)
    assert not np.array_equal(field_1, env.field)


@pytest.mark.parametrize("k, n_reports, drones, robots", [(3, 2, 1, 1), (5, 5, 2, 2), (7, 4, 4, 4)])
def test_observation_and_state_sizes(k, n_reports, drones, robots):
    config = small_config(obs_window=k, n_reports=n_reports, n_drones=drones, n_robots=robots)
    env = WeedEnv(config)
    obs, _ = env.reset(seed=0)
    expected = 6 + 4 * k * k + 3 * n_reports
    assert observation_size(k, n_reports) == expected
    for agent in env.agents:
        assert obs[agent].shape == (expected,)
        assert env.observation_space(agent).contains(obs[agent])
    map_size = min(config.state_map_size, 16)
    assert env.state().shape == (state_size(drones + robots, map_size),)
    assert env.state().shape == (3 * (drones + robots) + 3 * map_size**2 + 1,)
    info = env.get_env_info()
    assert info["obs_shape"] == expected and info["n_actions"] == N_ACTIONS


def test_out_of_bounds_move_is_stay():
    env = WeedEnv(small_config(n_drones=1, n_robots=1))
    env.reset(seed=0)
    env.positions[:] = [[0, 0], [15, 15]]
    battery = env.battery.copy()
    env.step({"drone_0": UP, "robot_0": DOWN})
    np.testing.assert_array_equal(env.positions, [[0, 0], [15, 15]])
    env.step({"drone_0": LEFT, "robot_0": RIGHT})
    np.testing.assert_array_equal(env.positions, [[0, 0], [15, 15]])
    np.testing.assert_array_equal(env.battery, battery)
    np.testing.assert_array_equal(env.path_length, [0, 0])


def test_empty_battery_allows_only_stay():
    env = WeedEnv(small_config(n_drones=1, n_robots=1))
    _, infos = env.reset(seed=0)
    env.battery[:] = 0.0
    env.positions[:] = [[5, 5], [8, 8]]
    np.testing.assert_array_equal(env.action_masks(), [[1, 0, 0, 0, 0, 0]] * 2)
    env.step({"drone_0": RIGHT, "robot_0": UP})
    np.testing.assert_array_equal(env.positions, [[5, 5], [8, 8]])


def test_drone_mask_never_allows_spray():
    env = WeedEnv(small_config())
    _, infos = env.reset(seed=0)
    for agent in env.drones:
        np.testing.assert_array_equal(infos[agent]["action_mask"], [1, 1, 1, 1, 1, 0])
    for agent in env.robots:
        assert infos[agent]["action_mask"][SPRAY] == 1
        assert infos[agent]["action_mask"].dtype == np.int8


def test_truncation_at_max_steps():
    env = WeedEnv(small_config(max_steps=5))
    env.reset(seed=0)
    for step in range(5):
        _, _, terminations, truncations, infos = env.step({a: STAY for a in env.agents})
        expected = step == 4
        assert all(v == expected for v in truncations.values())
        assert not any(terminations.values())
    assert env.agents == []
    episode = infos["robot_0"]["episode"]
    assert episode["truncated"] and not episode["terminated"]
    assert episode["end_reasons"] == ["max_steps"]


def test_termination_when_robot_batteries_run_out():
    config = small_config(robot_battery=3.0, max_steps=50, n_drones=1, n_robots=2)
    env = WeedEnv(config)
    env.reset(seed=0)
    env.positions[1:] = [[0, 0], [0, 1]]
    for step in range(3):
        _, _, terminations, truncations, infos = env.step(
            {"drone_0": STAY, "robot_0": DOWN, "robot_1": DOWN}
        )
    assert all(terminations.values())
    assert not any(truncations.values())
    assert infos["robot_0"]["episode"]["end_reasons"] == ["robots_depleted"]
    assert env.agents == []


def test_reward_is_computed_once_per_step(monkeypatch):
    calls = []

    def fake_reward(weeds_sprayed=0, unnecessary_sprays=0, config=None):
        calls.append((weeds_sprayed, unnecessary_sprays))
        return -0.1

    monkeypatch.setattr(weed_env, "calculate_reward", fake_reward)
    env = WeedEnv(small_config(max_steps=6))
    env.reset(seed=0)
    for step in range(1, 7):
        _, rewards, _, _, infos = env.step({a: STAY for a in env.agents})
        assert len(calls) == step
        assert set(rewards.values()) == {-0.1}
        assert all(info["team_reward"] == -0.1 for info in infos.values())


def test_reward_uses_marl_reward_config():
    from marl.reward import RewardConfig

    env = WeedEnv(small_config(reward=RewardConfig(step_penalty=-0.5)))
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step({a: STAY for a in env.agents})
    assert set(rewards.values()) == {-0.5}


def test_npy_field_loading_and_crop():
    grid = np.load(GRID)
    env = WeedEnv(WeedEnvConfig(grid_path=str(GRID)))
    env.reset(seed=0)
    assert env.field.shape == (64, 64)
    np.testing.assert_array_equal(env.field, grid)
    assert env.n_weeds == int((grid == WEED).sum())

    cropped = WeedEnv(WeedEnvConfig(grid_path=str(GRID), crop_size=32, crop_origin=(10, 20)))
    cropped.reset(seed=0)
    np.testing.assert_array_equal(cropped.field, grid[10:42, 20:52])

    random_crop = WeedEnv(WeedEnvConfig(grid_path=str(GRID), crop_size=16))
    random_crop.reset(seed=3)
    first = random_crop.field.copy()
    random_crop.reset(seed=3)
    np.testing.assert_array_equal(first, random_crop.field)


def test_config_from_dict():
    config = WeedEnvConfig.from_dict(
        {"grid_size": 32, "n_drones": 1, "reward": {"step_penalty": -1.0}, "crop_origin": [0, 0]}
    )
    assert config.grid_size == 32 and config.reward.step_penalty == -1.0
    with pytest.raises(ValueError):
        WeedEnvConfig.from_dict({"bilinmeyen": 1})


def test_drone_detection_writes_explored_map():
    env = WeedEnv(small_config(n_drones=1, n_robots=1, detection_radius=1))
    env.reset(seed=0)
    env.positions[0] = [5, 5]
    env.step({"drone_0": STAY, "robot_0": STAY})
    assert env.drone_explored[0, 4:7, 4:7].all()
    np.testing.assert_array_equal(env.drone_weed[0, 4:7, 4:7], env.field[4:7, 4:7] == WEED)


def test_robot_knowledge_only_comes_from_drone_reports():
    env = WeedEnv(small_config(n_drones=1, n_robots=2, detection_radius=1, max_steps=40))
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    while env.agents:
        env.step(random_actions(env, rng))
        scanned = env.drone_explored.any(axis=0)
        for j in range(2):
            # Robot yalnızca drone'ların taradığı hücreleri bilebilir
            assert not (env.robot_explored[j] & ~scanned).any()
            assert not (env.robot_known_weed[j] & ~env.weed_mask).any()


def test_robot_observation_has_no_weed_info_outside_reports():
    env = WeedEnv(small_config(n_drones=1, n_robots=1, detection_radius=0, obs_window=3))
    env.reset(seed=0)
    env.positions[:] = [[0, 0], [15, 15]]
    env.robot_explored[:] = False
    env.robot_known_weed[:] = False
    env.field[14:16, 14:16] = WEED
    env.weed_mask = env.field == WEED
    obs, _, _, _, _ = env.step({"drone_0": STAY, "robot_0": STAY})
    robot = obs["robot_0"]
    window = robot[6 : 6 + 4 * 9].reshape(4, 3, 3)
    assert window[0].sum() == 0  # bilinen ot yok
    assert robot[6 + 36 :].sum() == 0  # rapor yok


def test_state_is_fixed_size_and_bounded():
    env = WeedEnv(WeedEnvConfig(grid_path=str(GRID), max_steps=10))
    env.reset(seed=0)
    s = env.state()
    assert s.shape == env.state_space.shape
    assert env.state_space.contains(s)
    assert s[-1] == 0.0


def test_render_ansi_and_rgb():
    env = WeedEnv(small_config(), render_mode="ansi")
    env.reset(seed=0)
    lines = env.render().splitlines()
    assert len(lines) == 17 and all(len(line) == 16 for line in lines[:16])
    rgb = WeedEnv(small_config(), render_mode="rgb_array")
    rgb.reset(seed=0)
    assert rgb.render().shape == (128, 128, 3)


def test_epymarl_helpers_are_consistent():
    env = WeedEnv(small_config())
    obs, _ = env.reset(seed=0)
    assert len(env.get_obs()) == 4
    np.testing.assert_array_equal(env.get_obs_agent(3), obs["robot_1"])
    assert env.get_avail_agent_actions(0) == [1, 1, 1, 1, 1, 0]
    assert env.get_state().shape == (env.get_state_size(),)
    assert env.get_total_actions() == 6
