import math

import numpy as np
import pygame as pg

from cnn_model.constants import FEATURES
from game.constants import (
    BLACK,
    CELL_HEIGHT,
    CELL_WIDTH,
    COLS,
    FONT,
    FONT_HEIGHT,
    FPS,
    LIGHT_GRAY,
    MOVES_PER_SECOND,
    ROWS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
)
from game.fruit import Fruit
from game.snake import Direction, Snake

successes, failures = pg.init()
print("{0} successes and {1} failures".format(successes, failures))


class Game:
    def reset(self) -> np.ndarray:
        self.snake.reset()
        self.fruit.generate_fruit(self.snake.positions)
        self.score = 0
        return self.game_state()

    def __init__(self, use_graphics=True):
        self.snake = Snake()
        self.fruit = Fruit(self.snake.positions)
        self.clock = pg.time.Clock()
        self.highscore = 0
        self.score = 0

        self.use_graphics = use_graphics
        if self.use_graphics:
            self.window = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.window.fill(WHITE)
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

    def game_state_nn(self) -> np.ndarray:
<<<<<<< HEAD
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
=======
    def game_state_cnn(self) -> np.ndarray:
        state = np.zeros((ROWS, COLS, FEATURES), dtype=np.float_)

        # Snake (tail)
        for pos in self.snake.positions[1:]:
            (x, y) = pos
            state[y][x][0] = 0.5

        # Snake (head)
        (x, y) = self.snake.positions[0]
        state[y][x][0] = 1.0

        # Fruit (goal)
        (x, y) = self.fruit.position
        state[y][x][1] = 1.0
>>>>>>> ff5240b (Add CNN model)
        return state

    def distance(self) -> float:
        head = self.snake.positions[0]
        goal = self.fruit.position
        return math.sqrt((head[0] - goal[0]) ** 2 + (head[1] - goal[1]) ** 2)

    def run_action(self, action: Direction):
        if action == Direction.UP:
            self.snake.turn(Direction.UP)
        elif action == Direction.LEFT:
            self.snake.turn(Direction.LEFT)
        elif action == Direction.RIGHT:
            self.snake.turn(Direction.RIGHT)
        elif action == Direction.DOWN:
            self.snake.turn(Direction.DOWN)

        game_over = False

        is_alive = self.snake.move()
        if not is_alive:
            game_over = True
        elif self.snake.positions[0] == self.fruit.position:
            self.score += 1
            if self.score > self.highscore:
                self.highscore = self.score
            self.fruit.generate_fruit(self.snake.positions)
        else:
            self.snake.pop_end()

        if self.use_graphics:
            self.render()

        return (self.game_state(), self.score, game_over, self.distance())

    def play_game(self):
        running = True
        iteration = 0
        while running:
            self.clock.tick(FPS)
            direction = None

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        direction = Direction.LEFT
                    elif event.key == pg.K_UP:
                        direction = Direction.UP
                    elif event.key == pg.K_RIGHT:
                        direction = Direction.RIGHT
                    elif event.key == pg.K_DOWN:
                        direction = Direction.DOWN

            iteration += 1
            if direction is not None or iteration >= FPS / MOVES_PER_SECOND:
                _, _, game_over, _ = self.run_action(
                    direction if direction is not None else self.snake.direction
                )
                iteration = 0
                if game_over:
                    self.reset()
                    self.render()

    def render(self):
        self.draw_grid()
        self.draw_score()
        self.fruit.render(self.window)
        self.snake.render(self.window)
        pg.display.update()

    def draw_grid(self):
        for y in range(ROWS):
            for x in range(COLS):
                rect = pg.Rect(
                    CELL_WIDTH * x,
                    CELL_HEIGHT * y + FONT_HEIGHT,
                    CELL_WIDTH,
                    CELL_HEIGHT,
                )
                color = WHITE if (x + y) % 2 else LIGHT_GRAY
                pg.draw.rect(self.window, color, rect)

    def draw_score(self):
        rect = pg.Rect(
            0,
            0,
            SCREEN_WIDTH,
            FONT_HEIGHT,
        )
        pg.draw.rect(self.window, WHITE, rect)
        score_text = FONT.render("Score: {}".format(self.score), True, BLACK)
        high_score = FONT.render("High-score: {}".format(self.highscore), True, BLACK)
        self.window.blit(score_text, (5, 5))
        self.window.blit(high_score, (SCREEN_WIDTH - 5 - high_score.get_width(), 5))


if __name__ == "__main__":
    game = Game()
    game.play_game()
