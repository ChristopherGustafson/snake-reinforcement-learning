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

    """
    Game state defined as 12-feature vector:
    0. danger up 1 or 2 tiles
    1. danger down 1 or 2 tiles
    2. danger left 1 or 2 tiles
    3. danger right 1 or 2 tiles
    4. moving up
    5. moving down
    6. moving left
    7. moving right
    8. fruit up
    9. fruit down
    10. fruit left
    11. fruit right
    """

    def game_state(self) -> np.ndarray:
        state = np.zeros(12)
        # ***** Check for dangers
        # snake head y = 0 => danger up
        if self.snake.positions[0][1] == 0 or self.snake.positions[0][1] == 1:
            state[0] = 1
        # snake head y = max_y => danger down
        elif (
            self.snake.positions[0][1] == rows - 1
            or self.snake.positions[0][1] == rows - 2
        ):
            state[1] = 1
        # snake head x = 0 => danger left
        if self.snake.positions[0][0] == 0 or self.snake.positions[0][0] == 1:
            state[2] = 1
        # snake head y = 0 => danger right
        if (
            self.snake.positions[0][1] == cols - 1
            or self.snake.positions[0][1] == cols - 2
        ):
            state[3] = 1

        # ***** Check for directions
        if self.snake.direction == Direction.UP:
            state[4] = 1
        elif self.snake.direction == Direction.DOWN:
            state[5] = 1
        elif self.snake.direction == Direction.LEFT:
            state[6] = 1
        elif self.snake.direction == Direction.RIGHT:
            state[7] = 1

        # ***** Check for fruit location
        # snake y > fruit y => fruit is above snake
        if self.snake.positions[0][1] > self.fruit.position[1]:
            state[8] = 1
        # snake y < fruit y => fruit is below snake
        elif self.snake.positions[0][1] < self.fruit.position[1]:
            state[9] = 1
        # snake x < fruit x => fruit is left of snake
        elif self.snake.positions[0][0] < self.fruit.position[0]:
            state[10] = 1
        # snake x > fruit x => fruit is right of snake
        elif self.snake.positions[0][0] > self.fruit.position[0]:
            state[11] = 1
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
