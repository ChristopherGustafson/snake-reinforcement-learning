import sys
import pygame as pg
pg.init()

clock = pg.time.Clock()
clock.tick(1)
# Game Constants
rows = cols = 20

width, height = 400, 400
window = pg.display.set_mode((width,height))
window.fill((255,255,255))

for y in range(rows):
    for x in range(cols):
        rect = pg.Rect(width/cols * x, height/rows * y, width/cols, height/rows)
        color = (255,255,255) if x+y%2 else (0,0,0)
        pg.draw.rect(window, color, rect)
        pg.display.update()

while True:
    clock.tick(1)
    pass