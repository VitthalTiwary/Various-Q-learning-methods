import numpy as np
import random
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

grid_size = 13
goal_state = (9, 12)
initial_state = (0, 1)
kill_state = [  (0,0), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
                (1,0), (1,2), (1,6), (1,10), (1,12),
                (2,0), (2,2), (2,3), (2,4), (2,6), (2,7), (2,8), (2,10), (2,12),
                (3,0), (3,6), (3,12),
                (4,0), (4,2), (4,3), (4,4), (4,6), (4,8), (4,10), (4,11), (4,12),
                (5,0), (5,2), (5,8), (5,10), (5,12),
                (6,0), (6,1), (6,2), (6,4), (6,5), (6,6), (6,8), (6,9), (6,10), (6,12),
                (7,0), (7,4), (7,6), (7,12),
                (8,0), (8,2), (8,4), (8,6), (8,7), (8,8), (8,9), (8,10), (8,12),
                (9,0), (9,2), (9,10),
                (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10), (10,11), (10,12),
                (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7), (11,8), (11,9), (11,10), (11,11), (11,12),
                (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7), (12,8), (12,9), (12,10), (12,11), (12,12)]

reward_goal = 0
reward_other = -1
reward_kill = -20
gamma = 0.8
alpha = 0.1

epsilon = 0.1
max_steps_threshold = 500000

count = 0
counting = 0
number_of_episodes = 100
Episode = []

up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]
action_prob = 0.25  

global_initial_policy = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}

def find_first_occurrence_index(episode, target_tuple):
    try:
        index = next(i for i, value in enumerate(episode) if value[:2] == target_tuple)
        return index
    except StopIteration:
        return -1

def sum_rewards_from_index(episode, start_index):
    total_reward = 0
    for _, _, reward in episode[start_index:]:
        total_reward = gamma*total_reward + reward
    return total_reward

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
    else:
        return state

def generate_episode(policy, epsilon, counting):
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
        
        if steps > max_steps_threshold:
            # print(f"Episode aborted after exceeding {max_steps_threshold} steps.")
            return generate_episode(policy, epsilon, counting)
        
    print(f"Episode number {counting + 1} generated.")
    return episode

while count != number_of_episodes:
    Episode.append(generate_episode(global_initial_policy, 0.1, counting))
    counting += 1
    count += 1

def q_learning(distance, policy, skip):

    Q = np.zeros((len(actions), grid_size, grid_size))

    def update_policy(policy, Q):
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal_state:
                    continue

                best_action = np.argmax(Q[:, i, j])
                
                for a in actions:
                    if a == best_action:
                        policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                    else:
                        policy[(i, j)][a] = epsilon / len(actions)

    def max_state_value(state):
        i, j = state
        return max(Q[up][i, j], Q[down][i, j], Q[left][i, j], Q[right][i, j])


    for ep in range(distance):
        episode = Episode[ep + skip]
        # print(f"Episode {ep + 1}: Length = {len(episode)}")
        
        for state, action, reward in episode:
            next_state = get_next_state(state, action)
            Q[action][state] = Q[action][state] + alpha * ( reward + gamma * max_state_value(next_state) - Q[action][state] )

        # print(f"Updating policy after episode {ep + 1}...")

        update_policy(policy, Q)
    return Q, policy

def double_q_learning(distance, policy, skip):

    Q1 = np.zeros((len(actions), grid_size, grid_size))
    Q2 = np.zeros((len(actions), grid_size, grid_size))
    epsilon = 0.8

    def update_policy(policy, Q1, Q2, epsilon):
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal_state:
                    continue

                combined_q = Q1[:, i, j] + Q2[:, i, j]
                best_action = np.argmax(combined_q)

                for a in actions:
                    if a == best_action:
                        policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                    else:
                        policy[(i, j)][a] = epsilon / len(actions)

    for ep in range(distance):
        episode = Episode[ep + skip]
        # print(f"Episode {ep + 1}: Length = {len(episode)}")

        for state, action, reward in episode:
            next_state = get_next_state(state, action)

            if random.random() < 0.5:
                best_next_action = np.argmax(Q1[:, next_state[0], next_state[1]])
                Q1[action][state[0], state[1]] += alpha * (reward + gamma * Q2[best_next_action, next_state[0], next_state[1]] - Q1[action][state[0], state[1]])
            else:
                best_next_action = np.argmax(Q2[:, next_state[0], next_state[1]])
                Q2[action][state[0], state[1]] += alpha * (reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] - Q2[action][state[0], state[1]])

        update_policy(policy, Q1, Q2, epsilon)

    return ((Q1 + Q2)/2), policy

def triple_q_learning(distance, policy, skip):

    Q1 = np.zeros((len(actions), grid_size, grid_size))
    Q2 = np.zeros((len(actions), grid_size, grid_size))
    Q3 = np.zeros((len(actions), grid_size, grid_size))
    epsilon = 0.1

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

    for ep in range(distance):
        episode = Episode[ep + skip]
        # print(f"Episode {ep + 1}: Length = {len(episode)}")

        for state, action, reward in episode:
            next_state = get_next_state(state, action)

            # Randomly choose one of the three Q-tables for updating
            choice = random.choice([1, 2, 3])

            if choice == 1:
                best_next_action = np.argmax(Q1[:, next_state[0], next_state[1]])
                Q1[action][state[0], state[1]] += alpha * (reward + gamma * Q2[best_next_action, next_state[0], next_state[1]] - Q1[action][state[0], state[1]])
            elif choice == 2:
                best_next_action = np.argmax(Q2[:, next_state[0], next_state[1]])
                Q2[action][state[0], state[1]] += alpha * (reward + gamma * Q3[best_next_action, next_state[0], next_state[1]] - Q2[action][state[0], state[1]])
            else:
                best_next_action = np.argmax(Q3[:, next_state[0], next_state[1]])
                Q3[action][state[0], state[1]] += alpha * (reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] - Q3[action][state[0], state[1]])

        # Update the policy
        update_policy(policy, Q1, Q2, Q3, epsilon)

    return ((Q1 + Q2 + Q3)/3), policy

def quadruple_q_learning(distance, policy, skip):

    Q1 = np.zeros((len(actions), grid_size, grid_size))
    Q2 = np.zeros((len(actions), grid_size, grid_size))
    Q3 = np.zeros((len(actions), grid_size, grid_size))
    Q4 = np.zeros((len(actions), grid_size, grid_size))
    epsilon = 0.1

    def update_policy(policy, Q1, Q2, Q3, Q4, epsilon):
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

    for ep in range(distance):
        episode = Episode[ep + skip]
        # print(f"Episode {ep + 1}: Length = {len(episode)}")

        for state, action, reward in episode:
            next_state = get_next_state(state, action)

            q_choice = random.choice([1, 2, 3, 4])
            if q_choice == 1:
                best_next_action = np.argmax(Q1[:, next_state[0], next_state[1]])
                Q1[action, state[0], state[1]] += alpha * (reward + gamma * Q2[best_next_action, next_state[0], next_state[1]] - Q1[action, state[0], state[1]])
            elif q_choice == 2:
                best_next_action = np.argmax(Q2[:, next_state[0], next_state[1]])
                Q2[action, state[0], state[1]] += alpha * (reward + gamma * Q3[best_next_action, next_state[0], next_state[1]] - Q2[action, state[0], state[1]])
            elif q_choice == 3:
                best_next_action = np.argmax(Q3[:, next_state[0], next_state[1]])
                Q3[action, state[0], state[1]] += alpha * (reward + gamma * Q4[best_next_action, next_state[0], next_state[1]] - Q3[action, state[0], state[1]])
            else:
                best_next_action = np.argmax(Q4[:, next_state[0], next_state[1]])
                Q4[action, state[0], state[1]] += alpha * (reward + gamma * Q1[best_next_action, next_state[0], next_state[1]] - Q4[action, state[0], state[1]])

        update_policy(policy, Q1, Q2, Q3, Q4, epsilon)

    return ((Q1 + Q2 + Q3 + Q4)/4), policy

def optimal_policy_check(optimal_policy_recieved):
    episodes_for_check = 1000
    #print("number of episodes for checking optimal policy:", episodes_for_check)
    actions = [up, down, left, right]

    V = np.zeros((len(actions), grid_size, grid_size))

    policy = optimal_policy_recieved

    states_action_pair = []
    for row in range(grid_size):
        for col in range(grid_size):
            for a in actions:
                states_action_pair.append(((row, col), a))

    def generate_episode_for_check(policy):
        episode = []
        state = initial_state
        while state != goal_state:
            action = random.choices(actions, weights=policy[state])[0]
            next_state = get_next_state(state, action)
                
            if next_state == kill_state:
                reward = reward_kill
                next_state = initial_state
            else:
                reward = reward_goal if next_state == goal_state else reward_other
            
            episode.append((state, action, reward))
            state = next_state

        episode.append((state, action, reward_goal))  
        return episode

    avg_final_reward = 0
    avg_len_of_episode = 0

    for ep in range(episodes_for_check):
        episode = generate_episode_for_check(policy)
        #print(f"the legth of episode is: {len(episode)}")

        avg_len_of_episode = (ep*avg_len_of_episode + len(episode))/(ep + 1)

        for s_a in states_action_pair:
            ind = find_first_occurrence_index(episode, s_a)
            if ind != -1:
                total_reward = sum_rewards_from_index(episode, ind)
                #print(f"Sum of rewards from state {s_a} starting at index {ind} is {total_reward}.")
            else:
                total_reward = 0
                #print(f"State {s_a} is not in the episode.")
            s, a = s_a
            V[a][s] = (ep*V[a][s] + total_reward)/(ep + 1)
        
        final_reward = sum_rewards_from_index(episode, 0)
        #print(f"\nThe total reward gained from this episode is: {final_reward}\n")

        avg_final_reward = (avg_final_reward*ep + final_reward)/(ep + 1)
    #print(f"\nThe average reward gained from all episodes is: {avg_final_reward}\n")

    #print(V)

    return avg_final_reward, avg_len_of_episode

E_values = []

Q1_values = []
Q2_values = []
Q3_values = []
Q4_values = []

R_Q1_values = []
R_Q2_values = []
R_Q3_values = []
R_Q4_values = []

Avg_len_Q1_values = []
Avg_len_Q2_values = []
Avg_len_Q3_values = []
Avg_len_Q4_values = []

optimal_q_policy, optimal_double_q_policy, optimal_triple_q_policy, optimal_quadruple_q_policy = global_initial_policy, global_initial_policy, global_initial_policy, global_initial_policy

r = 11
m = 6

for e in range(r):
    print(e+1, ". Checking for ", m*(e+1), " episodes.\n")
    E_values.append(m*(e+1))

    print("Generating policies.")

    # print(f"\nFor {m*(e+1)} episodes the result of the Q1 policy is:")
    Q_single, optimal_q_policy = q_learning(m, optimal_q_policy, e*m)
    Q1_values.append(Q_single[0][0,0])

    # print(f"\nFor {m*(e+1)} episodes the result of the Q2 policy is:")
    Q_double, optimal_double_q_policy = double_q_learning(m, optimal_double_q_policy, e*m)
    Q2_values.append(Q_double[0][0,0])

    # print(f"\nFor {m*(e+1)} episodes the result of the Q3 policy is:")
    Q_triple, optimal_triple_q_policy = triple_q_learning(m, optimal_triple_q_policy, e*m)
    Q3_values.append(Q_triple[0][0,0])

    # print(f"\nFor {m*(e+1)} episodes the result of the Q4 policy is:")
    Q_quadruple, optimal_quadruple_q_policy = quadruple_q_learning(m, optimal_quadruple_q_policy, e*m)
    Q4_values.append(Q_quadruple[0][0,0])

    print("\nChecking policies acquired.\n")
    R_Q1, Ep_len_Q1 = optimal_policy_check(optimal_q_policy)
    print("Policy check complete for 1 Q.")
    R_Q2, Ep_len_Q2 = optimal_policy_check(optimal_double_q_policy)
    print("Policy check complete for 2 Q.")
    R_Q3, Ep_len_Q3 = optimal_policy_check(optimal_triple_q_policy)
    print("Policy check complete for 3 Q.")
    R_Q4, Ep_len_Q4 = optimal_policy_check(optimal_quadruple_q_policy)
    print("Policy check complete for 4 Q.\n")

    Avg_len_Q1_values.append(Ep_len_Q1)
    Avg_len_Q2_values.append(Ep_len_Q2)
    Avg_len_Q3_values.append(Ep_len_Q3)
    Avg_len_Q4_values.append(Ep_len_Q4)
    R_Q1_values.append(R_Q1)
    R_Q2_values.append(R_Q2)
    R_Q3_values.append(R_Q3)
    R_Q4_values.append(R_Q4)

print("\nData accumulated.\n")

R_values_list = [R_Q1_values, R_Q2_values, R_Q3_values, R_Q4_values]
labels = ['R_Q1', 'R_Q2', 'R_Q3', 'R_Q4']

# Plot scatter points and best-fit lines for each R_Q dataset
plt.figure(figsize=(10, 6))
for R_values, label in zip(R_values_list, labels):
    # Scatter plot for current R_Q dataset (no markers)
    plt.scatter(E_values, R_values, label=label)

    # Calculate the linear fit (best-fit line)
    slope, intercept = np.polyfit(E_values, R_values, 1)
    best_fit_line = np.array(E_values) * slope + intercept

    # Plot the best-fit line (solid line)
    plt.plot(E_values, best_fit_line, linestyle='-', label=f'{label} Best Fit')

# Add labels, title, legend, and grid
plt.xlabel('E_values')
plt.ylabel('R_Q values')
plt.title('Scatter Plot with Best Fit Lines for R_Q Values')
plt.legend()
plt.grid(True)
plt.savefig(f'Rewards_Plot_{r}x{m}.png', dpi=300)

# Show plot
plt.show()

# Plot for average lengths (no markers)
plt.plot(E_values, Avg_len_Q1_values, label='Avg Length Q1')
plt.plot(E_values, Avg_len_Q2_values, label='Avg Length Q2')
plt.plot(E_values, Avg_len_Q3_values, label='Avg Length Q3')
plt.plot(E_values, Avg_len_Q4_values, label='Avg Length Q4')

# Adding labels and legend
plt.xlabel('E values')
plt.ylabel('Average Length')
plt.title('Average Lengths Q1 to Q4 vs E values')
plt.legend()
plt.grid(True)
plt.savefig(f'Average_Length_Plot_{r}x{m}.png', dpi=300)

# Show plot
plt.show()

# Plot for Q values (no markers)
plt.plot(E_values, Q1_values, label='Q1 Values')
plt.plot(E_values, Q2_values, label='Q2 Values')
plt.plot(E_values, Q3_values, label='Q3 Values')
plt.plot(E_values, Q4_values, label='Q4 Values')

# Adding labels and legend
plt.xlabel('E values')
plt.ylabel('Q Values')
plt.title('Q1 to Q4 Values vs E values')
plt.legend()
plt.grid(True)
plt.savefig(f'Q_Value_Plot_{r}x{m}.png', dpi=300)

# Show plot
plt.show()