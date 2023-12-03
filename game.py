import random

class Board:
    def __init__(self):
        self.score = 0
        self.game_over = False
        self.board = [[0]*4 for _ in range(4)]
        self._add_new_tile()
        self._add_new_tile()
        self.last_added_tile = None
        self.merges_in_last_move = 0

    def print_board(self):
        # Iterates through each row of the board and prints it
        for row in self.board:
            print(' '.join(map(str, row)))

    def print_score(self):
        # Outputs the current score
        print(f"Score: {self.score}")

    def move_tiles(self, direction):
        if self.game_over:
            return False

        self.merges_in_last_move = 0

        original_board = [row[:] for row in self.board]
        if direction in ('u', 'd'):
            self._transpose_board()

        for i in range(4):
            if direction in ('u', 'l'):
                shifted_row = self._shift(self.board[i])
                self.board[i] = self._merge(shifted_row)
            else:  # 'd' or 'r'
                reversed_row = list(reversed(self.board[i]))
                shifted_row = self._shift(reversed_row)
                merged_row = self._merge(shifted_row)
                self.board[i] = list(reversed(merged_row))

        if direction in ('u', 'd'):
            self._transpose_board()

        if self.board != original_board:
            self._add_new_tile()
            if not self._moves_available():
                self.game_over = True
            return True
        return False

    def _transpose_board(self):
        # Converts rows to columns and columns to rows
        transposed = []
        for col_index in range(4):
            new_row = []
            for row in self.board:
                new_row.append(row[col_index])
            transposed.append(new_row)
        self.board = transposed

    def _shift(self, row):
        # Shifts non-zero elements to the left in a row
        shifted_row = []
        for value in row:
            if value != 0:
                shifted_row.append(value)
        # Append zeros to the end of the row to maintain its size
        while len(shifted_row) < 4:
            shifted_row.append(0)
        return shifted_row

    def _merge(self, row):
        # Merge adjacent tiles with the same value
        for i in range(3):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
                self.merges_in_last_move += 1
        return self._shift(row)  # Shift again to ensure tiles are properly aligned

    def _add_new_tile(self):
        # Adds a new tile (2 or 4) to a random empty position on the board
        empty_positions = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_positions.append((i, j))
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.board[i][j] = 4 if random.random() < 0.1 else 2
            self.last_added_tile = (i, j)

    def _moves_available(self):
        # Check if there are any moves available
        for i in range(4):
            for j in range(4):
                # Check for empty spot
                if self.board[i][j] == 0:
                    return True
                # Check for possible merges in the row
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return True
                # Check for possible merges in the column
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return True
        return False

# The main function for running the game
def main():
    game = BoardClass()
    move_commands = {'u': 'up', 'd': 'down', 'l': 'left', 'r': 'right'}

    while not game.game_over:
        game.print_board()
        game.print_score()
        move = input("Enter your move (u, d, l, r): ").lower()
        if move in move_commands:
            if not game.move_tiles(move):
                print("Move not possible. Try a different direction.")
        else:
            print("Invalid input. Please enter 'u', 'd', 'l', or 'r'.")
        if game.game_over:
            print("Game over! Your final score is:", game.score)