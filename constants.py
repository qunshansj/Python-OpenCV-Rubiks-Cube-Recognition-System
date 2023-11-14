python

class RubiksCube:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.color_placeholder = (150, 150, 150)
        self.locales = {
            'de': 'Deutsch',
            'hu': 'Hungarian',
            'fr': 'French',
            'en': 'English',
            'nl': 'Nederlands',
            'es': 'Spanish',
            'zh': '简体中文',
        }
        self.mini_sticker_area_tile_size = 14
        self.mini_sticker_area_tile_gap = 2
        self.mini_sticker_area_offset = 20
        self.sticker_area_tile_size = 30
        self.sticker_area_tile_gap = 4
        self.sticker_area_offset = 20
        self.sticker_contour_color = (36, 255, 12)
        self.calibrate_mode_key = 'c'
        self.switch_language_key = 'l'
        self.text_size = 18
        self.cube_palette = 'cube_palette'
        self.errors = {
            'incorrectly_scanned': 1,
            'already_solved': 2
        }
