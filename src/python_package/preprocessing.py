from python_package.Sudoku import Sudoku
class Preprocessing(Sudoku):
    def __init__(self):
        super().__init__()
        
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

    ## Find open tuples that have unique assignments
    def fix_tuples(open_tuples):
        # Create dictionary to count occurrences of each cell position with potential digits
        count_dict = {}
        # Count each cell's possible digits
        for tup in open_tuples:
            key = (tup[0], tup[1])  # Use cell position as key
            if key in count_dict:
                count_dict[key].append(tup)
            else:
                count_dict[key] = [tup]
        # Collect only those cell positions with a single possible digit
        fixed_tuples = []
        for key, value in count_dict.items():
            if len(value) == 1:
                fixed_tuples.extend(value)
        return fixed_tuples

    ## Use the previous functions to find and fill all trivially solvable cells
    def preprocess(self):
        # Process to reduce possible digits for each cell
        open_tuples = self.find_open_tuples(self.puzzle.board)
        fixed_tuples = self.fix_tuples(open_tuples)
        # Keep processing while there are cells with a single possible digit
        while len(fixed_tuples) != 0:
            for tuple in fixed_tuples:
                i, j, digit = tuple  # Extract row, column, and digit
                self.puzzle.board[i][j] = digit  # Set digit on the board
            # Recalculate possible digits after setting known digits
            open_tuples = self.find_open_tuples(self.puzzle.board)
            fixed_tuples = self.fix_tuples(open_tuples)
            #puzzle.show()  # Display the board state
        return 0