from pettingzoo.test import parallel_api_test

from sim.example_parallel_env import MAX_STEPS, ExampleParallelEnv, run_random_episode


def test_parallel_api():
    parallel_api_test(ExampleParallelEnv(), num_cycles=MAX_STEPS + 10)


def test_random_episode_runs_until_step_limit():
    assert run_random_episode(ExampleParallelEnv(), seed=42) == MAX_STEPS
