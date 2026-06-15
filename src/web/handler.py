import numpy as np
from playwright.sync_api import sync_playwright

from game import UNKNOWN


class WebHandler:
    def __init__(self, headless=False):
        self.url = 'https://minesweeper.online/start/3'
        self.reset_btn_id = 'top_area_face'
        self.headless = headless
        self.browser = None
        self.playwright = None
        self.page = None
        self.cached_tiles = set()
        self.outcomes = {
            'hd_type1': 1,
            'hd_type2': 2,
            'hd_type3': 3,
            'hd_type4': 4,
            'hd_type5': 5,
            'hd_type6': 6,
            'hd_type7': 7,
            'hd_type8': 8,
        }

    def open(self):
        if self.browser:
            return

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless
        )

        self.page = self.browser.new_page()
        self.page.goto(self.url)

    def close(self):
        if self.browser:
            self.browser.close()
            self.browser = None

        if self.playwright:
            self.playwright.stop()
            self.playwright = None

        self.page = None

    def create_grid(self, height=16, width=30):
        return np.full((height, width), UNKNOWN, dtype=np.int8)

    def reset_game(self):
        self._require_open_page()
        self.page.locator(f'#{self.reset_btn_id}').click()
        self.cached_tiles.clear()

    def update_grid_from_open_tiles(self, grid):
        self._require_open_page()
        tiles = self.page.locator('.hd_opened').all()

        for tile in tiles:
            x = int(tile.get_attribute('data-x'))
            y = int(tile.get_attribute('data-y'))
            coords = (x, y)

            if coords in self.cached_tiles:
                continue

            class_names = tile.get_attribute('class') or ''
            grid[y][x] = self._value_from_tile_classes(class_names)

            self.cached_tiles.add(coords)

        return grid

    def wait_after_move(self, delay_seconds):
        self._require_open_page()
        self.page.wait_for_timeout(int(delay_seconds * 1000))

    def game_state(self):
        self._require_open_page()

        face_classes = (
            self.page.locator(f'#{self.reset_btn_id}').get_attribute('class')
            or ''
        )
        normalized = face_classes.lower()

        if any(marker in normalized for marker in ('facedead', 'face_dead', 'dead')):
            return 'lost'
        if any(marker in normalized for marker in ('facewin', 'face_win', 'win')):
            return 'won'

        if self.page.locator('.hd_type10, .hd_type11, .hd_type12').count() > 0:
            return 'lost'

        return 'playing'

    def click_tile(self, x, y):
        self._require_open_page()
        self.page.locator(f'#cell_{x}_{y}').click()

    def _value_from_tile_classes(self, class_names):
        for class_name in class_names.split():
            if class_name in self.outcomes:
                return self.outcomes[class_name]
        return 0

    def _require_open_page(self):
        if not self.page:
            raise RuntimeError(
                'WebHandler.open() must be called before interacting with the page.'
            )
