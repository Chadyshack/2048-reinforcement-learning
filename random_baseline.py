from game import Board
import random

def evaluate_random_actions(num_games=1000):
    top_scores = []
    total_score = 0
    for game_num in range(num_games):
        game = Board()
        while not game.game_over:
            possible_moves = game.possible_moves()
            action = random.choice(possible_moves)
            game.move_tiles(action)
        total_score += game.score
        top_scores.append(game.highest_tile())
        if game_num % 100 == 0:
            print(f"Played {game_num} games...")
    avg_score = total_score / num_games
    print(f"Average Score with Random Actions: {avg_score}")
    return top_scores

scores = evaluate_random_actions()
