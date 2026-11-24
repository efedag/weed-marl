
from baselines.fixed_route import (
    FixedRoutePolicy,
    create_scan_route,
    move_towards,
)
from sim.example_parallel_env import RIGHT, DOWN, STAY


def test_route_covers_all_cells():
    route = create_scan_route(8)

    assert len(route) == 64
    assert len(set(route)) == 64


def test_route_moves_are_adjacent():
    route = create_scan_route(8)

    for current, target in zip(route, route[1:]):
        distance = (
            abs(current[0] - target[0])
            + abs(current[1] - target[1])
        )
        assert distance == 1


def test_move_towards():
    assert move_towards((0, 0), (0, 1)) == RIGHT
    assert move_towards((0, 0), (1, 0)) == DOWN
    assert move_towards((0, 0), (0, 0)) == STAY


def test_policy_returns_valid_action():
    policy = FixedRoutePolicy(grid_size=8)

    action = policy.act((0, 0))

    assert action in range(5)
