import numpy as np
import random

# Configuration
grid_size = 10
goal_state = (9, 9)
initial_state = (0, 0)
kill_state = [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
              (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
              (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)]

reward_goal = 0
reward_other = -1
reward_kill = -20
gamma = 0.8
alpha = 0.1

episodes = 50
epsilon = 0.1  # Initial epsilon value
epsilon_decay = 0.0  # Decrease epsilon by this value every 10 episodes
min_epsilon = 0.1  # Lower bound for epsilon
step_limit = 1000000  # Limit for the number of steps in an episode

# Actions
up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]

# Initialize three Q-tables
Q1 = np.zeros((len(actions), grid_size, grid_size))
Q2 = np.zeros((len(actions), grid_size, grid_size))
Q3 = np.zeros((len(actions), grid_size, grid_size))

# Policy initialization
policy = {(i, j): [0.25] * len(actions) for i in range(grid_size) for j in range(grid_size)}

def get_next_state(state, action):
    i, j = state
    if action == up and i > 0:
        return (i - 1, j)
    elif action == down and i < grid_size - 1:
        return (i + 1, j)
    elif action == left and j > 0:
        return (i, j - 1)
    elif action == right and j < grid_size - 1:
        return (i, j + 1)
    return state

def generate_episode(policy):
    episode = []
    state = initial_state
    steps = 0

    while state != goal_state:
        action = random.choices(actions, weights=policy[state])[0]
        next_state = get_next_state(state, action)

        if next_state in kill_state:
            reward = reward_kill
            next_state = initial_state  # Restart from the initial state
        else:
            reward = reward_goal if next_state == goal_state else reward_other

        episode.append((state, action, reward))
        state = next_state
        steps += 1

        if steps > step_limit:
            print(f"Episode aborted after exceeding {step_limit} steps.")
            return generate_episode(policy)

    return episode

def update_policy(policy, Q1, Q2, Q3, epsilon):
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                continue

            combined_q = Q1[:, i, j] + Q2[:, i, j] + Q3[:, i, j]
            best_action = np.argmax(combined_q)

            for a in actions:
                if a == best_action:
                    policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                else:
                    policy[(i, j)][a] = epsilon / len(actions)

def max_action_value(Q, state):
    """Returns the maximum action value from Q for a given state."""
    i, j = state
    return max(Q[up][i, j], Q[down][i, j], Q[left][i, j], Q[right][i, j])

for ep in range(episodes):
    episode = generate_episode(policy)
    print(f"Episode {ep + 1}: Length = {len(episode)}")

    for state, action, reward in episode:
        next_state = get_next_state(state, action)

        # Randomly choose one of the three Q-tables for updating
        choice = random.choice([1, 2, 3])


        if choice == 1:
            best_next_action = np.argmax(Q2[:, next_state[0], next_state[1]] + Q3[:, next_state[0], next_state[1]])
            Q1[action][state[0], state[1]] += alpha * (reward + gamma * Q2[best_next_action, next_state[0], next_state[1]] - Q1[action][state[0], state[1]])
        elif choice == 2:
            best_next_action = np.argmax(Q1[:, next_state[0], next_state[1]] + Q3[:, next_state[0], next_state[1]])
            Q2[action][state[0], state[1]] += alpha * (reward + gamma * Q3[best_next_action, next_state[0], next_state[1]] - Q2[action][state[0], state[1]])
        else:
            best_next_action = np.argmax(Q1[:, next_state[0], next_state[1]] + Q2[:, next_state[0], next_state[1]])
            Q3[action][state[0], state[1]] += alpha * (reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] - Q3[action][state[0], state[1]])


    # Update the policy
    update_policy(policy, Q1, Q2, Q3, epsilon)

    # Decrease epsilon every 10 episodes
    if (ep + 1) % 10 == 0 and epsilon > min_epsilon:
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        print(f"Epsilon decayed to {epsilon:.2f}")

print("ENDING")

print("\nOptimal policy after all episodes:")
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i}, {j}): {policy[(i, j)]}")

np.set_printoptions(precision=2)
print("\nFinal Q1-values ((r,c), a):\n", Q1, "\n")
print("\nFinal Q2-values ((r,c), a):\n", Q2, "\n")
print("\nFinal Q3-values ((r,c), a):\n", Q3, "\n")