import torch
import torch.nn as nn
import torch.optim as optim
import random
from game import Board
import numpy as np
import torch.nn.functional as F

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
    
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * 16, 1024)
        self.dense2 = nn.Linear(1024, 4)

    import torch.nn.functional as F

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense2(x)

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
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
q_network = DQN()
optimizer = optim.Adam(q_network.parameters(), lr=5e-5)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(capacity=50000)
num_episodes = 20000
batch_size = 64
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = .9999
action_indices = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
gamma = 0.99

# Loop through batches
for episode in range(num_episodes):

    if episode % 100 == 0:
        print("Ran another 100 episodes...")

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
        if len(replay_buffer.buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            # Split batch into separate components, convert actions to indices
            states, actions, rewards, next_states, dones = zip(*batch)
            actions_modified = [action_indices[action] for action in actions]

            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions_modified)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)

            # Compute current Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1))

            # Compute next Q values
            next_q_values = q_network(next_states).max(1)[0]
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

    # Decay epsilon
    epsilon *= epsilon_decay

# TEMPORARY
def evaluate_model(model, num_games=1000):
    total_score = 0
    for game_num in range(num_games):
        game = Board()
        state = game.get_normalized_flattened_board()
        while not game.game_over:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values = model(state_tensor)
                
                # Get possible moves and mask action values for only possible actions
                possible_moves = game.possible_moves()
                masked_action_values = torch.full(action_values.shape, float('-inf'))
                for move in possible_moves:
                    index = action_indices[move]
                    masked_action_values[0][index] = action_values[0][index]
                
                # Choose the best action among the possible ones
                action = torch.argmax(masked_action_values).item()
                chosen_action = list(action_indices.keys())[list(action_indices.values()).index(action)]

            game.move_tiles(chosen_action)
            state = game.get_normalized_flattened_board()
        total_score += game.score
        #print(f"Game {game_num + 1}: Score = {game.score}")

    avg_score = total_score / num_games
    print(f"Average Score: {avg_score}")

# Call the evaluation function
evaluate_model(q_network)