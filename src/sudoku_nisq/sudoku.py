from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sudoku import Sudoku as pysudoku
from sudoku_py import SudokuGenerator as sudokupy
from sudoku_nisq.quantum import ExactCoverQuantumSolver
import os
import csv

class Sudoku():
    def __init__(self, grid_size=2, sudopy=True, num_missing_cells=6, pysudo=False, difficulty=0.4, seed=100, file_path='data/my_puzzle.csv'):
        self.grid_size = grid_size
        self.board_size = self.grid_size*self.grid_size
        self.total_cells = self.board_size * self.board_size
        self.difficulty = difficulty
        self.num_missing_cells = num_missing_cells
        self.file_path = file_path
        
        # Optionally use 
        #   py-sudoku: allows to generate puzzles from a seed.
        #   sudoku-py: allows to generate puzzles by number of blank cells.
        if pysudo is True:
            self.puzzle = pysudoku(self.grid_size,seed=seed).difficulty(self.difficulty)
            # print(self.puzzle.board)
        if sudopy is True:
            puzzle = sudokupy(board_size=self.board_size)
            cells_to_remove = self.num_missing_cells
            puzzle.generate(cells_to_remove=cells_to_remove)
            puzzle.board_exchange_values({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9})
            self.puzzle = puzzle
            # print(self.puzzle.board)

        self.open_tuples = self.find_open_tuples()
        self.pre_tuples = self.find_preset_tuples()
        self._init_quantum()
    
    def custom_board(self,board):
        self.puzzle.board = board
    
    
    def plot(self,title=None):
        """# Create an instance of the SudokuPuzzle class
            sudoku = SudokuPuzzle(...)

            # Plot the Sudoku grid
            fig = sudoku.plot(title="My Sudoku Puzzle")

            # To display the plot in an interactive environment
            fig.show()

            # Alternatively, to save the plot to a file
            fig.savefig("sudoku_plot.png") 
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_xlim(0, self.board_size)
        ax.set_ylim(0, self.board_size)
        
        minor_ticks = range(0, self.board_size + 1)
        major_ticks = range(0, self.board_size + 1, int(self.board_size**0.5))
        
        for tick in minor_ticks:
            ax.plot([tick, tick], [0, self.board_size], 'k', linewidth=0.5)
            ax.plot([0, self.board_size], [tick, tick], 'k', linewidth=0.5)
        
        for tick in major_ticks:
            ax.plot([tick, tick], [0, self.board_size], 'k', linewidth=3)
            ax.plot([0, self.board_size], [tick, tick], 'k', linewidth=3)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add numbers to the grid
        for (i, j, value) in self.pre_tuples:
            if value == 0:  # Check if the value is zero
                continue  # Skip the rest of the loop for this iteration
            ax.text(j + 0.5, self.board_size - 0.5 - i, str(value),
                    ha='center', va='center', fontsize=100/self.board_size)

            
        if title:
            plt.title(title, fontsize=20)
        
        plt.close(fig)

        return fig
    
    def _init_quantum(self,simple=True,pattern=False):
        self.quantum = ExactCoverQuantumSolver(self,simple=simple,pattern=pattern)
    
    def find_preset_tuples(self):
        preset_tuples = []
        for i in range(self.grid_size*self.grid_size):  # Loop over each row
            for j in range(self.grid_size*self.grid_size):  # Loop over each column in the row
                element = self.puzzle.board[i][j]
                if element is not None: # Check if the cell is pre-filled
                    preset_tuples.append((i,j,element)) # Store pre-filled cell as tuple
        return preset_tuples

    ## Find open cells and store them in tuples
    def find_open_tuples(self):
        open_tuples = []
        for i in range(self.grid_size*self.grid_size):  # Loop over each row
            for j in range(self.grid_size*self.grid_size):  # Loop over each column in the row
                element = self.puzzle.board[i][j]
                if element is None or element == 0: # Check if the cell is empty
                    digits = list(range(1, self.grid_size*self.grid_size +1)) # Possible digits for the cell
                    # Discard digits based on the column constraint
                    for p in range(self.grid_size*self.grid_size):
                        if self.puzzle.board[p][j] is not None and self.puzzle.board[p][j] != 0 and self.puzzle.board[p][j] in digits:
                            digits.remove(self.puzzle.board[p][j])
                    # Discard digits based on the row constraint
                    for q in range(self.grid_size*self.grid_size):
                        if self.puzzle.board[i][q] is not None and self.puzzle.board[i][q] != 0 and self.puzzle.board[i][q] in digits:
                            digits.remove(self.puzzle.board[i][q])
                    # Discard digits based on the subfield
                    subgrid_row_start = self.grid_size * (i // self.grid_size)
                    subgrid_col_start = self.grid_size * (j // self.grid_size)
                    for x in range(subgrid_row_start, subgrid_row_start + self.grid_size):
                        for y in range(subgrid_col_start, subgrid_col_start + self.grid_size):
                            if self.puzzle.board[x][y] is not None and self.puzzle.board[x][y] != 0 and self.puzzle.board[x][y] in digits:
                                digits.remove(self.puzzle.board[x][y])

                    # Store a tuple for each remaining possibility for the given cell
                    for digit in digits:
                        open_tuples.append((i, j, digit))
        return open_tuples