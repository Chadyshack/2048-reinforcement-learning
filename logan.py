import torch
import torch.nn as nn
import torch.optim as optim
import random
from game import Board
import numpy as np

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_episode(self, episode):
        for transition in episode:
            self.push(*transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Declare all variables and objects for training
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
q_network = QNetwork().to(device)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(capacity=10000)
num_episodes = 1000
batch_size = 100
epsilon = 1.0
epsilon_end = 0.1
epsilon_decay = 200
action_indices = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
gamma = 0.9

# Loop through episodes
for episode in range(num_episodes):
    if episode % 100 == 0:
        print("Episode:", episode)

    game = Board()
    state = game.get_normalized_flattened_board()
    episode_transitions = []

    while not game.game_over:
        possible_moves = game.possible_moves()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.uniform() < epsilon:
            with torch.no_grad():
                action_values = q_network(state_tensor)
                masked_action_values = torch.full(action_values.shape, float('-inf')).to(device)
                for action in possible_moves:
                    index = action_indices[action]
                    masked_action_values[0][index] = action_values[0][index]
                action = torch.argmax(masked_action_values).item()
        else:
            action = action_indices[random.choice(possible_moves)]

        game.move_tiles(action)
        reward = game.merges_in_last_move
        next_state = game.get_normalized_flattened_board()
        done = game.game_over
        episode_transitions.append((state, action, reward, next_state, done))
        state = next_state

    # Calculate returns
    G = 0
    returns = []
    for state, action, reward, next_state, done in reversed(episode_transitions):
        G = reward + gamma * G
        returns.insert(0, G)

    # Update the replay buffer with the episode
    replay_buffer.push_episode(list(zip([t[0] for t in episode_transitions],
                                        [t[1] for t in episode_transitions],
                                        returns,
                                        [t[3] for t in episode_transitions],
                                        [t[4] for t in episode_transitions])))

    # Update the network
    if len(replay_buffer.buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        states, actions, returns, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
        max_next_q_values = q_network(next_states).max(1)[0].detach()
        max_next_q_values[dones] = 0.0
        expected_q_values = returns.unsqueeze(1)

        loss = criterion(current_q_values, expected_q_values)

       
