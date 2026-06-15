from playwright.sync_api import sync_playwright


class WebHandler:
    def __init__(self):
        self.url = 'https://minesweeper.online/start/3'
        self.reset_btn_id = 'top_area_face'
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
        }

    def open(self):
        if self.browser:
            return

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=False
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

    def reset_game(self):
        self._require_open_page()
        self.page.locator(f'#{self.reset_btn_id}').click()
        self.cached_tiles.clear()

    def update_grid_from_open_tiles(self, grid):
        self._require_open_page()
        self.page.wait_for_selector('.hd_opened')
        tiles = self.page.locator('.hd_opened').all()

        for tile in tiles:
            x = int(tile.get_attribute('data-x'))
            y = int(tile.get_attribute('data-y'))
            coords = (x, y)

            if coords in self.cached_tiles:
                continue

            hd = tile.get_attribute('class').split()[-1]
            if hd in self.outcomes:
                grid[y][x] = self.outcomes[hd]

            self.cached_tiles.add(coords)

        return grid

    def click_tile(self, x, y):
        self._require_open_page()
        self.page.locator(f'#cell_{x}_{y}').click()

    def _require_open_page(self):
        if not self.page:
            raise RuntimeError(
                'WebHandler.open() must be called before interacting with the page.'
            )
