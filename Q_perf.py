import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_q_table(filename):
    try:
        with open(filename, "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully!")
        return q_table
    except FileNotFoundError:
        print("File not found.")
        return None

def plot_heatmap(q_table):
    # Convert Q-table into a 2D array for the heatmap
    data = [values for values in q_table.values()]
    data_array = np.array(data)

    plt.imshow(data_array, cmap='hot', interpolation='nearest')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.colorbar(label='Q-value')
    plt.title('Heatmap of Q-Table')
    plt.show()

# Load the Q-table
q_table = load_q_table("q_table.pickle")

# Check if the Q-table is loaded
if q_table is not None:
    # Print the Q-table data
    for state, actions in q_table.items():
        print(f"State: {state}, Actions: {actions}")

    # Plot the heatmap
    plot_heatmap(q_table)
