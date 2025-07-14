import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

# Configuration
grid_size = 10
goal_state = (9, 9)
initial_state = (0, 0)
kill_state = [(0,4),(0,5),(0,6),(0,7),(0,8),(1,4),(1,5),(1,6),(1,7),(1,8), (3,1),(4,1),(5,1),(6,1),(7,1),
              (3,2),(4,2),(5,2),(6,2),(7,2), (6,4),(6,5),(6,6),(6,7),(6,8),(7,4),(7,5),(7,6),(7,7),(7,8)]

reward_goal = 0
reward_other = 1
reward_kill = -20
gamma = 0.8
alpha = 0.1

episodes = 1000
epsilon = 0.1


up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]
action_prob = 0.25  

total_steps = 0
total_steps_arr = []
total_reward_arr = []

V = np.zeros((len(actions), grid_size, grid_size))

policy = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}

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
    total_reward = 0

    while state != goal_state:
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
        total_reward += reward

    return episode, steps, total_reward

def update_policy(policy, V):
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                continue

            best_action = np.argmax(V[:, i, j])
            
            for a in actions:
                if a == best_action:
                    policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                else:
                    policy[(i, j)][a] = epsilon / len(actions)

def max_state_value(state):
    i, j = state
    return max(V[up][i, j], V[down][i, j], V[left][i, j], V[right][i, j])


for ep in range(episodes):
    episode, steps, rewards = generate_episode(policy)
    # print(f"Episode {ep + 1}: {episode}")

    total_steps += steps

    total_steps_arr.append(total_steps)
    total_reward_arr.append(rewards)
    
    for state, action, reward in episode:
        next_state = get_next_state(state, action)
        V[action][state] = V[action][state] + alpha * ( reward + gamma * max_state_value(next_state) - V[action][state] )

    print(f"Updating policy after episode {ep + 1}...")

    update_policy(policy, V)

print("ENDING")

print("\nOptimal policy after all episodes:")
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i}, {j}): {policy[(i, j)]}")

print("\nFinal Q-values ((r,c), a):\n")
print(V, "\n")

print(f"Total number of steps gone through for each episode: {total_steps_arr}")
print(f"Total reward acquired at each episode: {total_reward_arr}")

plt.figure(figsize=(10, 6))
plt.plot(total_steps_arr, total_reward_arr, marker='o', color='b', linestyle='-', linewidth=1, markersize=3, label='Total Reward')
plt.title("Total Steps vs. Total Reward per Episode")
plt.xlabel("Total Steps")
plt.ylabel("Total Reward")
plt.legend()
plt.grid()
plt.show()