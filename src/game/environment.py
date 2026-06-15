from collections import deque
import random

import numpy as np


UNKNOWN = -1


class MinesweeperEnv:
    def __init__(
        self,
        height=16,
        width=30,
        mines=99,
        first_click_safe=True,
        first_click_safe_radius=1,
        seed=None,
    ):
        self.height = height
        self.width = width
        self.mines = mines
        self.first_click_safe = first_click_safe
        self.first_click_safe_radius = first_click_safe_radius
        self.rng = random.Random(seed)
        self.mine_grid = None
        self.number_grid = None
        self.visible_grid = None
        self.done = False
        self.won = False
        self.started = False
        self.reset()

    @property
    def action_size(self):
        return self.height * self.width

    def reset(self):
        self.mine_grid = np.zeros((self.height, self.width), dtype=bool)
        self.number_grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.visible_grid = np.full(
            (self.height, self.width),
            UNKNOWN,
            dtype=np.int8,
        )
        self.done = False
        self.won = False
        self.started = False

        if not self.first_click_safe:
            self._place_mines()

        return self.observation()

    def observation(self):
        return self.visible_grid.copy()

    def valid_action_mask(self):
        return (self.visible_grid.reshape(-1) == UNKNOWN).copy()

    def step(self, action):
        if self.done:
            return self.observation(), 0.0, True, {
                'invalid': True,
                'reason': 'game is already done',
            }

        x, y = self.action_to_coords(action)
        if not self._in_bounds(x, y):
            return self.observation(), -5.0, False, {
                'invalid': True,
                'reason': 'action out of bounds',
            }

        if self.visible_grid[y, x] != UNKNOWN:
            return self.observation(), -2.0, False, {
                'invalid': True,
                'reason': 'tile is already open',
            }

        if not self.started:
            if self.first_click_safe:
                self._place_mines(safe_cell=(x, y))
            self.started = True

        if self.mine_grid[y, x]:
            self.visible_grid[y, x] = 9
            self.done = True
            self.won = False
            return self.observation(), -25.0, True, {
                'hit_mine': True,
                'won': False,
                'opened_count': 0,
            }

        opened_count = self._reveal_from(x, y)
        reward = 1.0 + (0.1 * max(0, opened_count - 1))

        if self._is_won():
            self.done = True
            self.won = True
            reward += 25.0

        return self.observation(), reward, self.done, {
            'hit_mine': False,
            'won': self.won,
            'opened_count': opened_count,
        }

    def action_to_coords(self, action):
        action = int(action)
        y = action // self.width
        x = action % self.width
        return x, y

    def coords_to_action(self, x, y):
        return int(y) * self.width + int(x)

    def _place_mines(self, safe_cell=None):
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if not self._is_safe_start_cell(x, y, safe_cell)
        ]

        if self.mines > len(candidates):
            raise ValueError('Too many mines for the requested board settings.')

        for x, y in self.rng.sample(candidates, self.mines):
            self.mine_grid[y, x] = True

        self._calculate_numbers()

    def _is_safe_start_cell(self, x, y, safe_cell):
        if safe_cell is None:
            return False

        safe_x, safe_y = safe_cell
        return (
            abs(x - safe_x) <= self.first_click_safe_radius
            and abs(y - safe_y) <= self.first_click_safe_radius
        )

    def _calculate_numbers(self):
        self.number_grid.fill(0)

        for y in range(self.height):
            for x in range(self.width):
                if self.mine_grid[y, x]:
                    self.number_grid[y, x] = 9
                    continue

                self.number_grid[y, x] = sum(
                    self.mine_grid[ny, nx]
                    for nx, ny in self._neighbors(x, y)
                )

    def _reveal_from(self, start_x, start_y):
        opened_count = 0
        queue = deque([(start_x, start_y)])

        while queue:
            x, y = queue.popleft()
            if not self._in_bounds(x, y):
                continue
            if self.visible_grid[y, x] != UNKNOWN:
                continue
            if self.mine_grid[y, x]:
                continue

            self.visible_grid[y, x] = self.number_grid[y, x]
            opened_count += 1

            if self.number_grid[y, x] == 0:
                for nx, ny in self._neighbors(x, y):
                    if self.visible_grid[ny, nx] == UNKNOWN:
                        queue.append((nx, ny))

        return opened_count

    def _is_won(self):
        open_tiles = np.count_nonzero(self.visible_grid != UNKNOWN)
        return open_tiles == self.action_size - self.mines

    def _neighbors(self, x, y):
        for ny in range(max(0, y - 1), min(self.height, y + 2)):
            for nx in range(max(0, x - 1), min(self.width, x + 2)):
                if nx == x and ny == y:
                    continue
                yield nx, ny

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
