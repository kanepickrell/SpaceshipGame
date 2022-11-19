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

TRASHCAN = py.image.load(os.path.join("assets", "trash.png"))

#BACKGROUND = py.image.load(os.path.join("assets", "Oahu2.png"))
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
        #py.draw.rect(window, (0, 255,0), (self.x_pos, self.y_pos, 50, 50), width=0)
        window.blit(self.ship_image, (self.x_pos, self.y_pos))

    def get_width(self):
        return self.ship_image.get_width()
        
    def get_height(self):
        return self.ship_image.get_height()


class Player(Ship):
    def __init__(self, x_pos, y_pos, health=100) -> None:
        super().__init__(x_pos, y_pos, health)
        self.ship_image = YELLOW_SPACE_SHIP
        self.laser_image = BLUE_SPACE_LASER
        self.mask = py.mask.from_surface(self.ship_image)
        self.max_health = 100

class EnemyShip(Ship):
    COLOR_MAP = {

        "red": (RED_SPACE_SHIP, RED_SPACE_LASER),
        "blue": (BLUE_SPACE_SHIP, BLUE_SPACE_LASER),
        "green": (GREEN_SPACE_SHIP, GREEN_SPACE_LASER),

    }

    def __init__(self, x_pos, y_pos, color,health=100) -> None:
        super().__init__(x_pos, y_pos, health)
        self.ship_image, self.laser_image = self.COLOR_MAP[color]
        self.mask = py.mask.from_surface(self.ship_image)

    def move(self, vel):
        self.y_pos += vel


def main():
    run = True
    FPS = 60
    level = 0
    lives = 5
    player_vel = 5
    enemy_vel = 2

    lost = False
    lost_count = 0
    
    enemies = []
    wave_length = 5

    main_font = py.font.SysFont('comicsans', 35)
    lost_font = py.font.SysFont('comicsans', 65)
    clock = py.time.Clock()

#   s = Ship(300, 200)
    p = Player(300, 200)

    def redraw_window():
        WIN.blit(BACKGROUND_SCALED, (0, 0))
        lives_label = main_font.render(f"Lives: {lives}", 1, (255, 255, 255))
        level_label = main_font.render(f"Level: {level}", 1, (255, 255, 255))
        WIN.blit(lives_label, (10, 10))
        WIN.blit(level_label, (WIDTH - level_label.get_width() - 10, 10))

        for enemy in enemies:
            enemy.draw(WIN)

        p.draw(WIN)

        if lost:
            lost_label = lost_font.render(f"You have lost the game.", 1, (255,0,0))
            WIN.blit(lost_label, (WIDTH/2 - lost_label.get_width()/2, 350))

        py.display.update()

    while run:
        clock.tick(FPS)
        redraw_window()

        if lives <= 0 or p.health <= 0:
            lost = True
            lost_count += 1

        if lost:
            if lost_count > FPS * 3:
                run = False
            else:
                continue

        if len(enemies) == 0:
            level += 1
            wave_length += 10
            for i in range(wave_length):
                enemy = EnemyShip(random.randrange(50, WIDTH - 100), 
                                  random.randrange(-1500, -50), 
                                  random.choice(["red", "green", "blue"]))
                enemies.append(enemy)

        for event in py.event.get():
            if event.type == py.QUIT:
                run = False

        keys = py.key.get_pressed()
        if keys[py.K_a] and p.x_pos - player_vel > 0: # moves left
            p.x_pos -= player_vel
        if keys[py.K_d] and p.x_pos + player_vel + p.get_width() < WIDTH:
            p.x_pos += player_vel
        if keys[py.K_w] and p.y_pos - player_vel > 0: # moves down
            p.y_pos -= player_vel
        if keys[py.K_s]and p.y_pos + player_vel + p.get_height() < HEIGHT:  # move up
            p.y_pos += player_vel 

        for enemy in enemies[:]:
            enemy.move(enemy_vel)
            if enemy.y_pos + enemy.get_height() > HEIGHT:
                lives -= 1
                enemies.remove(enemy)

main()




