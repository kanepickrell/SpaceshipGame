import random

class Agent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1
        self.exploration_decay_rate = 0.001
        
        # Initialize the agent's parameters
        # For example, learning rate, discount factor, exploration rate, etc.
    
    def get_current_state(self, player, enemies, lasers):

        grid = [[0 for _ in range(3)] for _ in range(3)]
        sector_width = 750 / 3
        sector_height = 750 / 3

        for enemy in enemies:
            # Check if enemy is within the visible game area
            if 0 <= enemy.x_pos < 750 and 0 <= enemy.y_pos < 750:
                sector_x = int(enemy.x_pos // sector_width)
                sector_y = int(enemy.y_pos // sector_height)

                # Clamp the sector indices to be within the grid
                sector_x = max(0, min(sector_x, 3 - 1))
                sector_y = max(0, min(sector_y, 3 - 1))

                # Increment the count in the corresponding sector
                grid[sector_y][sector_x] += 1

        state = {
            "Agent Position": (player.x_pos, player.y_pos),
            "Enemy Sectors": grid,    }
        
        return state
    
    def select_action(self, state):
        state_key = str(state)  # Convert the state to a string (or other hashable form)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2])  # Explore: random action
        else:
            # Exploit: choose the best action based on Q-table
            self.q_table.setdefault(state_key, [0, 0, 0])
            return self.q_table[state_key].index(max(self.q_table[state_key]))
    
    def reward_function(self, player, enemies, prev_health):
    # Initialize variables for calculating average distance and reward
        total_distance = 0
        count = 0
        reward = 0  # Initialize reward with a default value

        # Calculate the distance from the player to each enemy
        for enemy in enemies:
            shooter_x = player.x_pos
            shooter_y = player.y_pos
            enemy_x = enemy.x_pos
            enemy_y = enemy.y_pos
            distance = ((shooter_x - enemy_x)**2 + (shooter_y - enemy_y)**2)**0.5

            if distance < 250:
                total_distance += distance
                count += 1
                print(f"Distance to enemy: {distance}")

        # Calculate the average distance only for enemies within 250 units
        if count > 0:
            average_distance = total_distance / count
            reward = average_distance/10

        # Check for collision and adjust the reward
        if player.health <= 0:
            reward -= 100  # Large penalty for collision

        return reward

    
    def learn(self, state, action, reward, next_state):
        state_key = str(state)
        next_state_key = str(next_state)
        self.q_table.setdefault(state_key, [0, 0, 0])
        self.q_table.setdefault(next_state_key, [0, 0, 0])

        # Q-learning update rule
        old_value = self.q_table[state_key][action]
        next_max = max(self.q_table[next_state_key])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state_key][action] = new_value

        # Update exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)


# Other methods can be added for specific functionalities like saving/loading policies, updating exploration rates, etc.
