import torch
import torch.nn as nn
import torch.optim as optim
import random
from game import Board
import numpy as np
import torch.nn.functional as F
import socket

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class QNetworkSimple(nn.Module):
    def __init__(self):
        super(QNetworkSimple, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class QNetworkDropout(nn.Module):
    def __init__(self):
        super(QNetworkDropout, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class QNetworkConv(nn.Module):
    def __init__(self):
        super(QNetworkConv, self).__init__()
        self.conv_block = ConvBlock(input_dim=1, output_dim=32)  # Example value
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*4*32, 128)  # Adjust the size accordingly
        self.fc2 = nn.Linear(128, 4)  # Final layer for 4 decisions

    def forward(self, x):
        x = x.to(device)
        x = self.conv_block(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        self.input_dim = input_dim
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, kernel_size=2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, kernel_size=4, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input to [batch_size, channels, height, width]
        if len(x.shape) == 1:  # Unbatched input
            x = x.view(1, self.input_dim, 4, 4)  # Assuming input is a flattened 4x4 board
        elif len(x.shape) == 2:  # Batched input
            x = x.view(-1, self.input_dim, 4, 4)  # -1 for batch size

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        return torch.cat((x1, x2, x3, x4), dim=1)

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

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Declare all variables and objects for training
q_network = QNetworkConv().to(device)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(capacity=1000)
num_episodes = 1000
batch_size = 100
epsilon = 0.9
epsilon_end = 0.01
epsilon_decay = .999
action_indices = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
gamma = 0.99

if torch.cuda.is_available():
    print(f'{socket.gethostname()} has an available cuda GPU.')
    
if torch.backends.mps.is_available():
    print(f'{socket.gethostname()} has an available mps GPU.')

# Loop through batches
for episode in range(num_episodes):
    # Start a new game board and get the initial state
    game = Board()
    state = game.get_normalized_flattened_board()

    # Loop through steps until the game is over
    while not game.game_over:
        # Get the possible moves
        possible_moves = game.possible_moves()

        # Choose an action using epsilon-greedy
        if np.random.uniform() < epsilon:
            with torch.no_grad():
                # Get action values from the network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values = q_network(state_tensor)

                # Filter action values for only possible actions
                masked_action_values = torch.full(action_values.shape, float('-inf'))
                for action in possible_moves:
                    index = action_indices[action]
                    masked_action_values[0][index] = action_values[0][index]
                action = torch.argmax(masked_action_values).item()

                # Convert action index to action key
                for action_key, index in action_indices.items():
                    if index == action:
                        action = action_key
                        break
        # Otherwise choose a random action
        else:
            action = random.choice(possible_moves)

        # Move the tiles using the action and store the reward, state, and done
        game.move_tiles(action)
        reward = game.merges_in_last_move
        next_state = game.get_normalized_flattened_board()
        done = game.game_over
        replay_buffer.push(state, action, reward, next_state, done)

        # Sample and update network if buffer is large enough
        if len(replay_buffer.buffer) > batch_size * 10:
            batch = replay_buffer.sample(batch_size)
            # Split batch into separate components, convert actions to indices
            states, actions, rewards, next_states, dones = zip(*batch)
            actions_modified = [action_indices[action] for action in actions]

            # Convert to tensors
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions_modified).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.BoolTensor(dones).to(device)

            # Compute current Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1))

            # Compute next Q values
            next_q_values = q_network(next_states).max(1)[0]

            # Zero out Q values that will lead to done state
            next_q_values[dones] = 0.0

            # Compute target Q values
            target_q_values = rewards + (gamma * next_q_values)

            # Compute loss
            loss = criterion(current_q_values, target_q_values.unsqueeze(1))

            # Optimize the network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the state
        state = next_state

    # Update epsilon
    epsilon *= epsilon_decay

    # Display progress
    if episode % 100 == 0:
            print(f"Ran {episode} episodes...")

# temp
def evaluate_model(num_games=1000):
    top_scores = []
    total_score = 0
    for game_num in range(num_games):
        game = Board()
        state = game.get_normalized_flattened_board()
        while not game.game_over:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values = q_network(state_tensor)                
                possible_moves = game.possible_moves()
                masked_action_values = torch.full(action_values.shape, float('-inf'))
                for move in possible_moves:
                    index = action_indices[move]
                    masked_action_values[0][index] = action_values[0][index]                
                action = torch.argmax(masked_action_values).item()
                chosen_action = list(action_indices.keys())[list(action_indices.values()).index(action)]
            game.move_tiles(chosen_action)
            state = game.get_normalized_flattened_board()
        total_score += game.score
        top_scores.append(game.highest_tile())
        if game_num % 100 == 0:
            print(f"Played {game_num} games...")
    avg_score = total_score / num_games
    print(f"Average Score: {avg_score}")
    return top_scores

# temp
scores = evaluate_model()

# Save the model
def save_model(name):
    torch.save(q_network.state_dict(), name)
