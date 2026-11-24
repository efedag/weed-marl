
from baselines.single_robot import SingleRobotPolicy
from sim.example_parallel_env import ExampleParallelEnv, STAY


def test_drone_stays_inactive():
    env = ExampleParallelEnv()
    observations, _ = env.reset(seed=42)
    policy = SingleRobotPolicy()

    actions = policy.act(observations)

    assert actions["drone_0"] == STAY
    assert actions["ground_robot_0"] in range(5)
    env.close()


def test_single_robot_episode():
    env = ExampleParallelEnv(grid_size=8, max_steps=50)
    observations, _ = env.reset(seed=42)
    policy = SingleRobotPolicy(grid_size=8)

    drone_start = tuple(observations["drone_0"])
    visited = {tuple(observations["ground_robot_0"])}
    steps = 0

    while env.agents:
        actions = policy.act(observations)
        observations, _, _, _, _ = env.step(actions)

        assert tuple(observations["drone_0"]) == drone_start

        visited.add(tuple(observations["ground_robot_0"]))
        steps += 1

    assert steps == 50
    assert len(visited) > 1
    env.close()
