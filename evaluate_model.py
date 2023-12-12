from game import Board
import random
import torch
import torch.nn as nn

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

action_indices = {'u': 0, 'd': 1, 'l': 2, 'r': 3}

def evaluate_model(model, num_games=1000):
    model = torch.load(model)
    model.eval()
    top_scores = []
    total_score = 0
    for game_num in range(num_games):
        game = Board()
        state = game.get_normalized_flattened_board()
        while not game.game_over:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values = model(state_tensor)                
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

scores = evaluate_model("model.plt")
