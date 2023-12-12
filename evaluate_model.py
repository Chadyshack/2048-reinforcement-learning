from game import Board
import random
import torch

action_indices = {'u': 0, 'd': 1, 'l': 2, 'r': 3}

def evaluate_model(model, num_games=1000, model_name):
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
        if game_num % 100 == 0:
            print(f"Played {game_num} games...")
    avg_score = total_score / num_games
    print(f"Average Score: {avg_score}")

evaluate_model(1000, "model.plt")
