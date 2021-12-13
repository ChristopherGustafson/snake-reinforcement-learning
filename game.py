import pygame as pg

from constants import BLACK, WHITE, cell_height, cell_width, cols, height, rows, width
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


fruit = Fruit(rows, cols)

running = True

snake = Snake()
while running:
    clock.tick(1)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_DOWN:
                snake.turn(Direction.DOWN)
            elif event.key == pg.K_UP:
                snake.turn(Direction.UP)
            elif event.key == pg.K_RIGHT:
                snake.turn(Direction.RIGHT)
            elif event.key == pg.K_LEFT:
                snake.turn(Direction.LEFT)

    draw_grid()
    fruit.draw_fruit(window)

    snake.move(False)
    if snake.check_collision():
        snake.reset()

    snake.render(window)
    pg.display.update()
