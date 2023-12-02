
import random

class Board:
    def __init__(self):
        self.score = 0
        self.game_over = False
        initial_values = [0] * 14 + [2] * 2
        random.shuffle(initial_values)
        self.board = [initial_values[i:i + 4] for i in range(0, 16, 4)]
        self.last_added_tile = None

    def print_board(self):
        for row in self.board:
            print(' '.join(map(str, row)))

    def print_score(self):
        print(f"Score: {self.score}")

    def move_tiles(self, direction):
        if self.game_over:
            return False

        original_board = [row[:] for row in self.board]
        if direction in ('u', 'd'):
            self.board = [list(col) for col in zip(*self.board)]  # Transpose for vertical movement
        for i in range(4):
            if direction in ('u', 'l'):
                self.board[i] = self._merge(self._shift(self.board[i]))
            else:
                self.board[i] = list(reversed(self._merge(self._shift(reversed(self.board[i])))))
        if direction in ('u', 'd'):
            self.board = [list(col) for col in zip(*self.board)]  # Transpose back

        if self.board != original_board:
            self._add_new_tile()
            if not self._moves_available():
                self.game_over = True
            return True
        return False

    def _shift(self, line):
        return [i for i in line if i != 0] + [0] * (4 - sum(1 for i in line if i != 0))

    def _merge(self, line):
        for i in range(3):
            if line[i] == line[i + 1] and line[i] != 0:
                line[i] *= 2
                line[i + 1] = 0
                self.score += line[i]
        return self._shift(line)

    def _add_new_tile(self):
        empty_positions = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.board[i][j] = 4 if random.random() < 0.1 else 2  # 10% chance to add a 4
            self.last_added_tile = (i, j)

    def _moves_available(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return True
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return True
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return True
        return False

def main():
    game = Board()
    move_commands = {'u': 'up', 'w': 'up', 'd': 'down', 's': 'down', 'l': 'left', 'a': 'left', 'r': 'right', 'd': 'right'}

    while not game.game_over:
        game.print_board()
        game.print_score()
        move = input("Enter your move (u, d, l, r or w, a, s, d): ").lower()
        if move in move_commands:
            if not game.move_tiles(move_commands[move][0]):
                print("Move not possible. Try a different direction.")
        else:
            print("Invalid input. Please enter 'u', 'd', 'l', 'r', 'w', 'a', 's', or 'd'.")
        if game.game_over:
            print("Game over! Your final score is:", game.score)

if __name__ == "__main__":
    main()
