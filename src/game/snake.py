from enum import Enum

import pygame as pg

from game.constants import CELL_HEIGHT, CELL_WIDTH, COLS, FONT_HEIGHT, ROWS


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


STARTING_X = int(COLS / 2)
STARTING_Y = int(ROWS / 5)
STARTING_DIR = Direction.DOWN
HEAD_COLOR = (0, 0, 100)
SNAKE_COLOR = (0, 100, 0)


class Snake:
    tail_color = SNAKE_COLOR
    head_color = HEAD_COLOR

    def __init__(self):
        self.reset()

    def reset(self):
        self.positions = [(STARTING_X, STARTING_Y), (STARTING_X, STARTING_Y - 1)]
        self.direction = STARTING_DIR

    @staticmethod
    def is_opposite(direction1: Direction, direction2: Direction):
        return (direction1.value + direction2.value) % 2

    def turn(self, direction: Direction):
        if self.is_opposite(direction, self.direction) != 0:
            self.direction = direction

    def move(self) -> bool:
        (x, y) = self.positions[0]
        if self.direction is Direction.UP:
            if self.is_safe(x, y - 1):
                self.positions.insert(0, (x, y - 1))
                return True
        elif self.direction is Direction.DOWN:
            if self.is_safe(x, y + 1):
                self.positions.insert(0, (x, y + 1))
                return True
        elif self.direction is Direction.RIGHT:
            if self.is_safe(x + 1, y):
                self.positions.insert(0, (x + 1, y))
                return True
        elif self.direction is Direction.LEFT:
            if self.is_safe(x - 1, y):
                self.positions.insert(0, (x - 1, y))
                return True
        return False

    def pop_end(self):
        self.positions.pop()

    def is_safe(self, x, y):
        # Check collision with walls
        if x < 0 or x == COLS or y < 0 or y == ROWS:
            return False
        # Check collision with itself
        if (x, y) in self.positions[:-1]:
            return False
        return True

    def render(self, window):
        for i, pos in enumerate(self.positions):
            (x, y) = pos
            rect = pg.Rect(
                CELL_WIDTH * x, CELL_HEIGHT * y + FONT_HEIGHT, CELL_WIDTH, CELL_HEIGHT
            )
            pg.draw.rect(window, self.head_color if i == 0 else self.tail_color, rect)
