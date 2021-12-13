import pygame as pg
from constants import (
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
from fruit import Fruit
from snake import Direction, Snake

successes, failures = pg.init()
print("{0} successes and {1} failures".format(successes, failures))

clock = pg.time.Clock()
clock.tick(1)

window = pg.display.set_mode((width, height))
window.fill(WHITE)


def draw_grid():
    for y in range(rows):
        for x in range(cols):
            rect = pg.Rect(cell_width * x, cell_height * y, cell_width, cell_height)
            color = WHITE if (x + y) % 2 else BLACK
            pg.draw.rect(window, color, rect)


running = True

snake = Snake()
fruit = Fruit(snake.positions)

score = 0

iteration = 0

while running:
    clock.tick(FPS)

    did_turn = False
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_DOWN:
                snake.turn(Direction.DOWN)
                did_turn = True
            elif event.key == pg.K_UP:
                snake.turn(Direction.UP)
                did_turn = True
            elif event.key == pg.K_RIGHT:
                snake.turn(Direction.RIGHT)
                did_turn = True
            elif event.key == pg.K_LEFT:
                snake.turn(Direction.LEFT)
                did_turn = True

    if iteration >= FPS / MOVES_PER_SECOND or did_turn:
        head = snake.move()
        if head == fruit.position:
            score += 1
            fruit.generate_fruit(snake.positions)
        else:
            snake.pop_end()
        iteration = 0

    if snake.check_collision():
        snake.reset()

    draw_grid()
    fruit.render(window)
    snake.render(window)
    pg.display.update()
    iteration += 1
