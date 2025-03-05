class SudokuPuzzle:
    def __init__(self, board=None, grid_size=3):
        self.grid_size = grid_size              # Number of blocks per row/column (e.g., 3 for a 9x9 puzzle)
        self.board_size = grid_size * grid_size  # Total size of the board (e.g., 9 for a 9x9 puzzle)
        # The board is stored in the puzzle object.
        self.puzzle = type("Puzzle", (), {})()  
        if board:
            self.puzzle.board = [row[:] for row in board]
        else:
            self.puzzle.board = [[0] * self.board_size for _ in range(self.board_size)]

    def set_cell(self, i, j, value):
        self.puzzle.board[i][j] = value
    
    def is_correct(self):
        board = self.puzzle.board
        size = self.board_size
        # Check rows for duplicates
        for i in range(size):
            seen = set()
            for j in range(size):
                num = board[i][j]
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        # Check columns for duplicates
        for j in range(size):
            seen = set()
            for i in range(size):
                num = board[i][j]
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        # Check subgrids (blocks) for duplicates
        for block_row in range(0, size, self.grid_size):
            for block_col in range(0, size, self.grid_size):
                seen = set()
                for i in range(block_row, block_row + self.grid_size):
                    for j in range(block_col, block_col + self.grid_size):
                        num = board[i][j]
                        if num != 0:
                            if num in seen:
                                return False
                            seen.add(num)
        return True
    
    def find_empty(self):
        board = self.puzzle.board
        size = self.board_size
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def count_solutions(self):
        """
        Recursively count all complete solutions for the puzzle.
        """
        empty = self.find_empty()
        if not empty:
            return 1  # Found a complete solution
        i, j = empty
        count = 0
        for num in range(1, self.board_size + 1):
            self.set_cell(i, j, num)
            if self.is_correct():
                count += self.count_solutions()
            self.set_cell(i, j, 0)  # Backtrack
        return count

# Example usage:
if __name__ == '__main__':
    # 9x9 Sudoku board (0 represents an empty cell)
    sample_board_9x9 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 0, 0, 0, 0, 3],
        [4, 0, 0, 0, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 0, 0, 0, 0, 6],
        [0, 6, 0, 8, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    puzzle9 = SudokuPuzzle(board=sample_board_9x9, grid_size=3)
    print("Count of complete solutions (9x9):", puzzle9.count_solutions())
    
    # 4x4 Sudoku board (0 represents an empty cell)
    sample_board_4x4 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    # For a 4x4 puzzle, grid_size would be 2.
    puzzle4 = SudokuPuzzle(board=sample_board_4x4, grid_size=2)
    print("Count of complete solutions (4x4):", puzzle4.count_solutions())
