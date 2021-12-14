import math

import numpy as np
import pygame as pg

from game.constants import (
    BLACK,
    FPS,
    MOVES_PER_SECOND,
    WHITE,
    cell_height,
    cell_width,
    cols,
    height,
    rows,
    width,
)
from game.fruit import Fruit
from game.snake import Direction, Snake

successes, failures = pg.init()
print("{0} successes and {1} failures".format(successes, failures))


window = pg.display.set_mode((width, height))
window.fill(WHITE)


class Game:
    def reset(self):
        self.snake = Snake()
        self.fruit = Fruit(self.snake.positions)
        self.score = 0

    def __init__(self):
        self.clock = pg.time.Clock()
        self.reset()

    def game_state(self) -> np.ndarray:
        state = np.zeros((rows, cols, 3))
        for pos in self.snake.positions[1:]:
            (x, y) = pos
            state[min(y, rows - 1)][min(x, cols - 1)] = [1, 0, 0]
        (x, y) = self.snake.positions[0]
        state[min(y, rows - 1)][min(x, cols - 1)] = [0, 1, 0]
        (x, y) = self.fruit.position
        state[y][x] = [0, 0, 1]
        return state

    def distance(self) -> float:
        head = self.snake.positions[0]
        goal = self.fruit.position
        return math.sqrt((head[0] - goal[0]) ** 2 + (head[1] - goal[1]) ** 2)

    def run_action(self, action: Direction):
        if action == Direction.UP:
            self.snake.turn(Direction.UP)
        elif action == Direction.DOWN:
            self.snake.turn(Direction.DOWN)
        elif action == Direction.RIGHT:
            self.snake.turn(Direction.RIGHT)
        elif action == Direction.LEFT:
            self.snake.turn(Direction.LEFT)

        head = self.snake.move()
        if head == self.fruit.position:
            self.score += 1
            self.fruit.generate_fruit(self.snake.positions)
        else:
            self.snake.pop_end()

        self.draw_grid()
        self.fruit.render(window)
        self.snake.render(window)
        pg.display.update()

        return (
            self.game_state(),
            self.score,
            self.snake.check_collision(),
            self.distance(),
        )

    def play_game(self):
        running = True
        score = 0
        iteration = 0

        while running:
            self.clock.tick(FPS)

            did_turn = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_DOWN:
                        self.snake.turn(Direction.DOWN)
                        did_turn = True
                    elif event.key == pg.K_UP:
                        self.snake.turn(Direction.UP)
                        did_turn = True
                    elif event.key == pg.K_RIGHT:
                        self.snake.turn(Direction.RIGHT)
                        did_turn = True
                    elif event.key == pg.K_LEFT:
                        self.snake.turn(Direction.LEFT)
                        did_turn = True

            if iteration >= FPS / MOVES_PER_SECOND or did_turn:
                head = self.snake.move()
                if head == self.fruit.position:
                    score += 1
                    self.fruit.generate_fruit(self.snake.positions)
                else:
                    self.snake.pop_end()
                iteration = 0

            if self.snake.check_collision():
                self.snake.reset()

            self.draw_grid()
            self.fruit.render(window)
            self.snake.render(window)
            pg.display.update()
            iteration += 1

    def draw_grid(self):
        for y in range(rows):
            for x in range(cols):
                rect = pg.Rect(cell_width * x, cell_height * y, cell_width, cell_height)
                color = WHITE if (x + y) % 2 else BLACK
                pg.draw.rect(window, color, rect)


if __name__ == "__main__":
    game = Game()
    game.play_game()
