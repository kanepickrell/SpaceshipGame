import pygame as py 
import os
import random
import time
py.font.init()

# display window
WIDTH = 750
HEIGHT = 750
WIN = py.display.set_mode((WIDTH, HEIGHT))
py.display.set_caption("Space Fighter Game")

# porting the assets
PATH = "/Users/kanepickrel/repos/py_scripts/space game/assets"
RED_SPACE_SHIP = py.image.load(os.path.join("assets", "pixel_ship_red_small.png"))
BLUE_SPACE_SHIP = py.image.load(os.path.join("assets", "pixel_ship_blue_small.png"))
GREEN_SPACE_SHIP = py.image.load(os.path.join("assets", "pixel_ship_green_small.png"))
YELLOW_SPACE_SHIP = py.image.load(os.path.join("assets", "pixel_ship_yellow.png"))

RED_SPACE_LASER = py.image.load(os.path.join("assets", "pixel_laser_red.png"))
BLUE_SPACE_LASER = py.image.load(os.path.join("assets", "pixel_laser_blue.png"))
GREEN_SPACE_LASER = py.image.load(os.path.join("assets", "pixel_laser_green.png"))
YELLOW_SPACE_LASER = py.image.load(os.path.join("assets", "pixel_laser_yellow.png"))

BACKGROUND = py.image.load(os.path.join("assets", "background-black.png"))
BACKGROUND_SCALED = py.transform.scale(BACKGROUND, (WIDTH, HEIGHT))

#this is an ABSTACT CLASS used later for inheritance.
class Ship:
    def __init__(self, x_pos, y_pos, health=100) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.health = health
        self.ship_image = None
        self.laser_image = None
        self.lasers = []
        self.cool_down_counter = 0

    def draw(self, window):
        py.draw.rect(window, (0, 255,0), (self.x_pos, self.y_pos, 50, 50), width=0)


def main():
    run = True
    FPS = 60
    level = 1
    lives = 5
    main_font = py.font.SysFont('comicsans', 35)
    clock = py.time.Clock()

    s = Ship(300, 200)

    def redraw_window():
        WIN.blit(BACKGROUND_SCALED, (0, 0))
        lives_label = main_font.render(f"Lives: {lives}", 1, (255, 255, 255))
        level_lable = main_font.render(f"Level: {level}", 1, (255, 255, 255))
        WIN.blit(lives_label, (10, 10))
        WIN.blit(level_lable, (WIDTH - level_lable.get_width() - 10, 10))

        s.draw(WIN)

        py.display.update()

    while run:
        clock.tick(FPS)
        
        redraw_window()

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False
main()




