import random
from typing import List, Tuple

import pygame as pg

from game.constants import CELL_HEIGHT, CELL_WIDTH, COLS, FONT_HEIGHT, ROWS


class Fruit:
    def __init__(self, invalid_positions: List[Tuple[int, int]]) -> None:
        self.generate_fruit(invalid_positions)

    def _generate_random_position(self) -> Tuple[int, int]:
        return (
            random.randint(1, COLS - 1),
            random.randint(1, ROWS - 1),
        )

    def generate_fruit(self, invalid_positions: List[Tuple[int, int]]) -> None:
        self.position = self._generate_random_position()
        while self.position in invalid_positions:
            self.position = self._generate_random_position()

    def render(self, screen: pg.surface.Surface) -> None:
        fruit = pg.Rect(
            CELL_WIDTH * self.position[0],
            CELL_HEIGHT * self.position[1] + FONT_HEIGHT,
            CELL_WIDTH,
            CELL_HEIGHT,
        )
        pg.draw.ellipse(screen, (255, 0, 0), fruit)
