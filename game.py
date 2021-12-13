import pygame as pg

from fruit import Fruit

successes, failures = pg.init()
print("{0} successes and {1} failures".format(successes, failures))

clock = pg.time.Clock()
clock.tick(1)
# Game Constants
rows = cols = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

width, height = 400, 400
window = pg.display.set_mode((width, height))
window.fill(WHITE)


def draw_grid():
    cell_height = height / rows
    cell_width = width / cols
    for y in range(rows):
        for x in range(cols):
            rect = pg.Rect(cell_width * x, cell_height * y, cell_width, cell_height)
            color = WHITE if (x + y) % 2 else BLACK
            pg.draw.rect(window, color, rect)


fruit = Fruit(rows, cols)

running = True

while running:
    clock.tick(1)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    draw_grid()
    fruit.draw_fruit(window)
    pg.display.update()
