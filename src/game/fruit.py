import random
from typing import List, Tuple

import pygame as pg
from constants import cell_height, cell_width, cols, rows


class Fruit:
    def __init__(self, invalid_positions: List[Tuple[int, int]]) -> None:
        self.generate_fruit(invalid_positions)

    def _generate_random_position(self) -> Tuple[int, int]:
        return (
            random.randint(0, cols - 1),
            random.randint(0, rows - 1),
        )

    def generate_fruit(self, invalid_positions: List[Tuple[int, int]]) -> None:
        self.position = self._generate_random_position()
        while self.position in invalid_positions:
            self.position = self._generate_random_position()

    def render(self, screen: pg.surface.Surface) -> None:
        fruit = pg.Rect(
            cell_width * self.position[0],
            cell_height * self.position[1],
            cell_width,
            cell_height,
        )
        pg.draw.ellipse(screen, (255, 0, 0), fruit)
