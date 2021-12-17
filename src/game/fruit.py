import random
from typing import List, Tuple

import pygame as pg

from game.constants import CELL_HEIGHT, CELL_WIDTH, COLS, FONT_HEIGHT, ROWS


class Fruit:
    def __init__(self, invalid_positions: List[Tuple[int, int]]) -> None:
        """
        Initialize a new fruit

        :param invalid_positions: Positions where the fruit may not spawn (inside the snake)
        """
        self.generate_fruit(invalid_positions)

    def _generate_random_position(self) -> Tuple[int, int]:
        """
        Generate a random poisition inside the grid.
        """
        return (
            random.randint(1, COLS - 1),
            random.randint(1, ROWS - 1),
        )

    def generate_fruit(self, invalid_positions: List[Tuple[int, int]]) -> None:
        """
        Generate a new position for the fruit.

        :param invalid_positions: Positions where the fruit may not spawn (inside the snake)
        """
        self.position = self._generate_random_position()
        while self.position in invalid_positions:
            self.position = self._generate_random_position()

    def render(self, screen: pg.surface.Surface) -> None:
        """
        Render the fruit to the screen.

        :param screen: The pygame surface object to render the fruit on
        """
        fruit = pg.Rect(
            CELL_WIDTH * self.position[0],
            CELL_HEIGHT * self.position[1] + FONT_HEIGHT,
            CELL_WIDTH,
            CELL_HEIGHT,
        )
        pg.draw.ellipse(screen, (255, 0, 0), fruit)
