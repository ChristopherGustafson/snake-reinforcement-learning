from enum import Enum

import pygame as pg

from game.constants import cell_height, cell_width, cols, rows


class Direction(Enum):
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4


def opposite(direction1: Direction, direction2: Direction):
    if direction1.value + direction2.value == 5:
        return True
    return False


STARTING_X = int(cols / 2)
STARTING_Y = int(rows / 5)
STARTING_DIR = Direction.DOWN
SNAKE_COLOR = (0, 100, 0)


class Snake:
    def __init__(self):
        self.positions = [(STARTING_X, STARTING_Y)]
        self.direction = STARTING_DIR
        self.color = SNAKE_COLOR

    def reset(self):
        self.positions = [(STARTING_X, STARTING_Y)]
        self.direction = STARTING_DIR
        self.color = SNAKE_COLOR

    def turn(self, direction: Direction):
        # Only allow turning in non-opposite direction
        if opposite(self.direction, direction):
            return
        self.direction = direction

    def move(self):
        (x, y) = self.positions[0]
        if self.direction == Direction.UP:
            self.positions.insert(0, (x, y - 1))
        elif self.direction == Direction.DOWN:
            self.positions.insert(0, (x, y + 1))
        if self.direction == Direction.RIGHT:
            self.positions.insert(0, (x + 1, y))
        if self.direction == Direction.LEFT:
            self.positions.insert(0, (x - 1, y))
        return self.positions[0]

    def pop_end(self):
        self.positions.pop()

    def check_collision(self):
        # Check collision with walls
        (x, y) = self.positions[0]
        if x < 0 or x == cols or y < 0 or y == rows:
            return True
        # Check collision with itself
        if self.positions[0] in self.positions[1:]:
            return True
        return False

    def render(self, window):
        for pos in self.positions:
            (x, y) = pos
            rect = pg.Rect(cell_width * x, cell_height * y, cell_width, cell_height)
            pg.draw.rect(window, self.color, rect)
