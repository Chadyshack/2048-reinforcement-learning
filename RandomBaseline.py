from game import Board
import random

def evaluate_random_actions(num_games=10000):
    total_score = 0
    for game_num in range(num_games):
        game = Board()
        while not game.game_over:
            possible_moves = game.possible_moves()
            action = random.choice(possible_moves)  # Choose a random action from possible moves
            game.move_tiles(action)
        total_score += game.score
        if game_num % 100:
            print("Played another 100 games...")

    avg_score = total_score / num_games
    print(f"Average Score with Random Actions: {avg_score}")

# Call the random action evaluation function
evaluate_random_actions()
