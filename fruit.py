import random
from typing import Tuple

import pygame as pg


class Fruit:
    def __init__(
        self,
        rows: int,
        cols: int,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.generate_fruit()

    def generate_fruit(self) -> None:
        self.position: Tuple[int, int] = (
            random.randint(0, self.cols - 1),
            random.randint(0, self.rows - 1),
        )

    def draw_fruit(self, screen: pg.surface.Surface) -> None:
        screen_height = screen.get_height()
        screen_width = screen.get_width()
        cell_width = screen_width / self.cols
        cell_height = screen_height / self.rows
        fruit = pg.Rect(
            cell_width * self.position[0],
            cell_height * self.position[1],
            cell_width,
            cell_height,
        )
        pg.draw.ellipse(screen, (255, 0, 0), fruit)
