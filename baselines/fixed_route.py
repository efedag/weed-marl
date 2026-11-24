
"""Fixed scanning route baseline for the weed-marl project."""

from sim.example_parallel_env import (
    STAY, UP, DOWN, LEFT, RIGHT
)


def create_scan_route(grid_size):
    """Generate a snake-shaped route covering every grid cell."""
    route = []

    for row in range(grid_size):
        if row % 2 == 0:
            columns = range(grid_size)
        else:
            columns = range(grid_size - 1, -1, -1)

        for col in columns:
            route.append((row, col))

    return route


def move_towards(current, target):
    """Choose one movement action toward the target cell."""
    row, col = current
    target_row, target_col = target

    if row < target_row:
        return DOWN
    if row > target_row:
        return UP
    if col < target_col:
        return RIGHT
    if col > target_col:
        return LEFT

    return STAY



class FixedRoutePolicy:
    """Cyclic lawnmower scanning policy."""

    def __init__(self, grid_size=8):
        self.route = create_scan_route(grid_size)
        self.target_index = 0
        self.visited_targets = 0
        self.started = False

    def reset(self):
        self.target_index = 0
        self.visited_targets = 0
        self.started = False

    def act(self, observation):
        current = tuple(int(x) for x in observation)

        if not self.started:
            self.started = True
            self.target_index = (
                self.route.index(current) + 1
            ) % len(self.route)
            self.visited_targets = 1

        while self.visited_targets < len(self.route):
            target = self.route[self.target_index]

            if current != target:
                return move_towards(current, target)

            self.visited_targets += 1
            self.target_index = (
                self.target_index + 1
            ) % len(self.route)

        return STAY
