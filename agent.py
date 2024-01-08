import random

class Agent:
    def __init__(self):
        pass
        # Initialize the agent's parameters
        # For example, learning rate, discount factor, exploration rate, etc.

    def learn(self, state, action, reward, next_state):
        pass
        # Implement the learning process here
        # Update the knowledge or policy based on the reward received and the transition
    
    def get_current_state(self, player, grid, lasers):
        state = {
            "player_pos": (player.x_pos, player.y_pos),
            "player_health": player.health,
            "enemy_sectors": grid,    }
        return state
    
    def select_action(self, state):
    # Implement the logic to select an action based on the current state
    # This can be a random action or an action based on learned policies
    # return random action for now
        return random.choice([0, 1, 2])
    
    def reward_function(self, player, enemies, lasers):
        pass
        # Implement the reward function here
        # This function should return a reward value based on the current state of the game


# Other methods can be added for specific functionalities like saving/loading policies, updating exploration rates, etc.
