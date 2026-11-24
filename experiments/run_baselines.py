"""Reproducible baseline coverage experiments."""

import csv
import statistics
from pathlib import Path

from baselines.fixed_route import FixedRoutePolicy
from baselines.single_robot import SingleRobotPolicy
from sim.example_parallel_env import ExampleParallelEnv, STAY


GRID_SIZE = 8
MAX_STEPS = 50
SEEDS = range(10)

OUTPUT_DIR = Path("experiments/results")


def run_episode(method, seed):
    env = ExampleParallelEnv(
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS,
    )

    observations, _ = env.reset(seed=seed)

    if method == "fixed_route":
        policy = FixedRoutePolicy(grid_size=GRID_SIZE)
        active_agent = "drone_0"

    elif method == "single_robot":
        policy = SingleRobotPolicy(grid_size=GRID_SIZE)
        active_agent = "ground_robot_0"

    else:
        raise ValueError(f"Unknown method: {method}")

    policy.reset()

    visited = {
        tuple(int(x) for x in observations[active_agent])
    }

    steps = 0

    while env.agents and steps < MAX_STEPS:

        if method == "fixed_route":
            actions = {
                "drone_0": policy.act(
                    observations["drone_0"]
                ),
                "ground_robot_0": STAY,
            }
        else:
            actions = policy.act(observations)

        observations, rewards, terminated, truncated, infos = (
            env.step(actions)
        )

        steps += 1

        if active_agent in observations:
            visited.add(
                tuple(
                    int(x)
                    for x in observations[active_agent]
                )
            )

    env.close()

    total_cells = GRID_SIZE * GRID_SIZE
    coverage = len(visited) / total_cells * 100

    return {
        "method": method,
        "seed": seed,
        "grid_size": GRID_SIZE,
        "steps": steps,
        "visited_cells": len(visited),
        "total_cells": total_cells,
        "coverage_percent": round(coverage, 4),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    summaries = []

    methods = ["fixed_route", "single_robot"]

    for method in methods:

        method_results = []

        print(f"\nRunning: {method}")

        for seed in SEEDS:

            result = run_episode(method, seed)

            results.append(result)
            method_results.append(result["coverage_percent"])

            print(
                f"Seed {seed}: "
                f"{result['coverage_percent']:.2f}% coverage"
            )

        summary = {
            "method": method,
            "episodes": len(method_results),
            "mean_coverage": round(
                statistics.mean(method_results), 4
            ),
            "min_coverage": min(method_results),
            "max_coverage": max(method_results),
            "std_coverage": round(
                statistics.stdev(method_results), 4
            ),
        }

        summaries.append(summary)

        print("\nSummary:")
        print(f"Mean: {summary['mean_coverage']:.2f}%")
        print(f"Min: {summary['min_coverage']:.2f}%")
        print(f"Max: {summary['max_coverage']:.2f}%")
        print(f"Std: {summary['std_coverage']:.2f}")

    results_path = OUTPUT_DIR / "baseline_results.csv"

    with results_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "baseline_summary.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\nExperiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()