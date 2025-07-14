import numpy as np
import random

# Configuration
grid_size = 10
goal_state = (9, 9)
initial_state = (0, 0)
kill_state = [(0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
              (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
              (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
              (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
              (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
              (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)]

reward_goal = 0
reward_other = -1
reward_kill = -20
gamma = 0.8
alpha = 0.1

# Epsilon settings
initial_epsilon = 0.5
min_epsilon = 0.1
decay_rate = 0.1
decay_interval = 10
total_episodes = 1000
max_steps_threshold = 100000

# Actions
up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]

# Initialize four Q-tables
Q1 = np.zeros((len(actions), grid_size, grid_size))
Q2 = np.zeros((len(actions), grid_size, grid_size))
Q3 = np.zeros((len(actions), grid_size, grid_size))
Q4 = np.zeros((len(actions), grid_size, grid_size))

# Policy initialization: uniform random initially
policy = {(i, j): [0.25] * len(actions) for i in range(grid_size) for j in range(grid_size)}

def get_next_state(state, action):
    """Returns the next state based on the given action."""
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
    """Generates an episode based on the current policy."""
    episode = []
    state = initial_state
    steps = 0

    while state != goal_state and steps < max_steps_threshold:
        action = random.choices(actions, weights=policy[state])[0]
        next_state = get_next_state(state, action)

        if next_state in kill_state:
            reward = reward_kill
            next_state = initial_state
        else:
            reward = reward_goal if next_state == goal_state else reward_other

        episode.append((state, action, reward))
        state = next_state
        steps += 1

    return episode, steps

def update_policy(policy, Q1, Q2, Q3, Q4, epsilon):
    """Updates the policy based on the current Q values."""
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                continue

            combined_q = Q1[:, i, j] + Q2[:, i, j] + Q3[:, i, j] + Q4[:, i, j]
            best_action = np.argmax(combined_q)

            for a in actions:
                if a == best_action:
                    policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                else:
                    policy[(i, j)][a] = epsilon / len(actions)

epsilon = initial_epsilon

for ep in range(total_episodes):
    episode, steps = generate_episode(policy)
    
    if steps >= max_steps_threshold:
        print(f"Episode {ep + 1}: Aborting episode after {steps} steps")
    else:
        print(f"Episode {ep + 1}: Completed in {steps} steps")

    # Update one of Q1, Q2, Q3, or Q4 randomly
    for state, action, reward in episode:
        next_state = get_next_state(state, action)

        q_choice = random.choice([1, 2, 3, 4])
        if q_choice == 1:
            best_next_action = np.argmax((Q2[:, next_state[0], next_state[1]] +
                                          Q3[:, next_state[0], next_state[1]] +
                                          Q4[:, next_state[0], next_state[1]]) / 3)
            Q1[action, state[0], state[1]] += alpha * (
                reward + gamma * Q2[best_next_action, next_state[0], next_state[1]] -
                Q1[action, state[0], state[1]]
            )
        elif q_choice == 2:
            best_next_action = np.argmax((Q1[:, next_state[0], next_state[1]] +
                                          Q3[:, next_state[0], next_state[1]] +
                                          Q4[:, next_state[0], next_state[1]]) / 3)
            Q2[action, state[0], state[1]] += alpha * (
                reward + gamma * Q3[best_next_action, next_state[0], next_state[1]] -
                Q2[action, state[0], state[1]]
            )
        elif q_choice == 3:
            best_next_action = np.argmax((Q1[:, next_state[0], next_state[1]] +
                                          Q2[:, next_state[0], next_state[1]] +
                                          Q4[:, next_state[0], next_state[1]]) / 3)
            Q3[action, state[0], state[1]] += alpha * (
                reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] -
                Q3[action, state[0], state[1]]
            )
        else:
            best_next_action = np.argmax((Q1[:, next_state[0], next_state[1]] +
                                          Q2[:, next_state[0], next_state[1]] +
                                          Q3[:, next_state[0], next_state[1]]) / 3)
            Q4[action, state[0], state[1]] += alpha * (
                reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] -
                Q4[action, state[0], state[1]]
            )

    update_policy(policy, Q1, Q2, Q3, Q4, epsilon)

    if (ep + 1) % decay_interval == 0:
        epsilon = max(min_epsilon, epsilon - decay_rate)
        print(f"Epsilon decayed to: {epsilon:.4f} after episode {ep + 1}")

print("ENDING")

print("\nOptimal policy after all episodes:")
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i}, {j}): {policy[(i, j)]}")

np.set_printoptions(precision=2)
print("\nFinal Q1-values:\n", Q1)
print("\nFinal Q2-values:\n", Q2)
print("\nFinal Q3-values:\n", Q3)
print("\nFinal Q4-values:\n", Q4)