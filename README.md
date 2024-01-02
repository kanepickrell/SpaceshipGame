# SpaceshipGame

README.md for Space Fighter Game
# Description
Space Fighter Game is a fast-paced arcade-style game developed using Python and Pygame. In this game, players navigate a spaceship, avoiding and fighting against enemy ships using lasers. The game features increasing difficulty levels and a life system.
Part of a tutorial I followed, I designed several classes to create an emulation of Space Invaders. I give credit to this program from a YouTuber Tech with Tim and his walkthrough of Pygame. 

# Requirements
Python 3.x
Pygame Library
Installation
Ensure Python 3.x is installed on your system.
Install Pygame using pip:
Copy code
pip install pygame
Game Assets
The game requires several image assets for ships, lasers, and the background, stored in the assets directory. Ensure this directory is in the same folder as the game script.

# Features
Player spaceship controlled with keyboard.
Multiple enemy types with unique behaviors.
Collision detection for ships and lasers.
Health and life system for the player.
Progressive difficulty with increasing levels.
Gameplay
Use W, A, S, D keys to move the player's spaceship.
Press Space to shoot lasers.
Avoid enemy ships and lasers.
Destroy enemy ships to increase the level.
Game ends when player loses all lives or health.
# Classes
Laser: Manages the laser's position, movement, and collision.
Ship: Base class for all ships, handling their behavior and actions.
Player: Inherits from Ship, represents the player's character.
EnemyShip: Inherits from Ship, represents enemy characters.
Functions
main(): Main game loop handling gameplay mechanics.
main_menu(): Start screen and entry point of the game.
collide(obj1, obj2): Checks and handles collision between objects.
Running the Game
To run the game, execute the script in a Python environment:

Copy code
python space_fighter_game.py
Developer Contact
Creator: kane pickrel
Email: kanepickrell@gmail.com
GitHub: kanepickrell
