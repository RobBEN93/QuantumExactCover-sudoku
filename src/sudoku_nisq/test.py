from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sudoku import Sudoku as pysudoku
from sudoku_py import SudokuGenerator as sudokupy
from sudoku_nisq.quantum import ExactCoverQuantumSolver
import os
import csv

class Sudoku():
    def __init__(self, board=None, grid_size=2, sudopy=True, num_missing_cells=6, pysudo=False, difficulty=0.4, seed=100, file_path='data/my_puzzle.csv'):
        
        self.grid_size = grid_size
        self.board_size = self.grid_size*self.grid_size
        self.total_cells = self.board_size * self.board_size
        self.difficulty = difficulty
        self.num_missing_cells = num_missing_cells
        self.file_path = file_path
        
        # Optionally use 
        #   custom board as matrix
        #   py-sudoku: allows to generate puzzles from a seed.
        #   sudoku-py: allows to generate puzzles by number of blank cells.
        
        if board:
            self.puzzle = None
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
    
    def _init_quantum(self,simple=True,pattern=False,num_solutions = 1):
        self.quantum = ExactCoverQuantumSolver(self,simple=simple,pattern=pattern,num_solutions=num_solutions)
    
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
    
sudoku = Sudoku()
print(sudoku.puzzle.board)

# Import necessary packages and modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Integer, Column
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, Any, Optional, List
import warnings
import logging
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings to keep output clean (use with caution in production)
warnings.filterwarnings('ignore')

# Import custom modules for the Sudoku and quantum solver logic
from sudoku_nisq.q_sudoku import Sudoku
from sudoku_nisq.quantum import ExactCoverQuantumSolver

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
# Configure logging to capture info and error messages in a log file.
logging.basicConfig(
    filename='app.log',          # Log file name
    filemode='a',                # Append mode (use 'w' to overwrite on each run)
    level=logging.INFO,          # Set the logging level to capture INFO and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format with timestamps
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# SQLAlchemy Setup for Database Interaction
# ------------------------------------------------------------
# Create the base class for declarative class definitions.
Base = declarative_base()

# Define a SQLAlchemy model for the 'quantum_resources' table.
class QuantumResources(Base):
    """
    SQLAlchemy model representing the 'quantum_resources' table.
    
    Attributes:
        id: Primary key of the record.
        missing_cells: Number of empty cells in the Sudoku puzzle.
        num_qubits_simple_encoding: Number of qubits required using the simple encoding.
        mcx_gates_simple_encoding: Number of MCX gates required using the simple encoding.
        total_gates_simple_encoding: Total number of gates required using the simple encoding.
        num_qubits_pattern_encoding: Number of qubits required using the pattern encoding.
        mcx_gates_pattern_encoding: Number of MCX gates required using the pattern encoding.
        total_gates_pattern_encoding: Total number of gates required using the pattern encoding.
    """
    __tablename__ = 'quantum_resources'

    id = Column(Integer, primary_key=True)
    missing_cells = Column(Integer)
    
    num_qubits_simple_encoding = Column(Integer)
    mcx_gates_simple_encoding = Column(Integer)
    total_gates_simple_encoding = Column(Integer)

    num_qubits_pattern_encoding = Column(Integer)
    mcx_gates_pattern_encoding = Column(Integer)
    total_gates_pattern_encoding = Column(Integer)

# ------------------------------------------------------------
# Data Generation Class
# ------------------------------------------------------------
class GenData:
    """
    Class for generating Sudoku puzzles and estimating the quantum resources required for solving them.

    Attributes:
        size: Grid size (e.g., 2 for a 4x4 Sudoku).
        board_size: Total number of sub-grids (size * size).
        num_puzzles: Total number of puzzles to generate.
        db_url: Database connection string (optional).
        total_cells: Total number of cells in the Sudoku board.
        data_list: List that stores computed quantum resource data for each puzzle.
        engine: SQLAlchemy engine for database operations (if db_url is provided).
    """
    def __init__(self, grid_size: int = 2, num_puzzles: int = 1000, db_url: Optional[str] = None):
        self.size = grid_size
        self.board_size = grid_size * grid_size
        self.num_puzzles = num_puzzles
        self.db_url = db_url
        self.total_cells = self.board_size * self.board_size
        self.data_list: List[Dict[str, Any]] = []
        # If a database URL is provided, create an engine for database interactions.
        if db_url:
            self.engine = create_engine(self.db_url)
        else:
            self.engine = None

    def find_quantum_resources(self, sudoku: Sudoku):
        """
        Compute the quantum resources required for solving a given Sudoku puzzle.

        The method calculates resources using two encoding schemes:
            - Simple encoding
            - Pattern encoding

        Parameters:
            sudoku: A Sudoku instance representing the puzzle.

        Returns:
            A tuple containing:
                sim_num_qubits, sim_mcx_gates, sim_total_gates,
                pat_num_qubits, pat_mcx_gates, pat_total_gates
        """
        # Calculate resources using simple encoding
        circ = ExactCoverQuantumSolver(sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_mcx_gates, sim_total_gates  = circ.find_resources()
        # Calculate resources using pattern encoding
        circ = ExactCoverQuantumSolver(sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_mcx_gates, pat_total_gates = circ.find_resources()
        return sim_num_qubits, sim_mcx_gates, sim_total_gates, pat_num_qubits, pat_mcx_gates, pat_total_gates

    def generate_data(self, num_empty_cells: int = None, num_cells_range: tuple = (1, 8)) -> pd.DataFrame:
        """
        Generate multiple Sudoku puzzles and compute the quantum resources for each puzzle.

        You can specify a fixed number of empty cells (num_empty_cells) or a range (num_cells_range).

        Parameters:
            num_empty_cells: Optional; specific number of missing cells for each puzzle.
            num_cells_range: Tuple indicating the range of missing cells if num_empty_cells is not specified.

        Returns:
            A pandas DataFrame containing the computed quantum resource data.
        """
        # Validate the input parameters for number of empty cells
        if num_empty_cells is not None:
            if not isinstance(num_empty_cells, int) or num_empty_cells < 0:
                raise ValueError("num_empty_cells must be a non-negative integer.")
        if not (isinstance(num_cells_range, tuple) and len(num_cells_range) == 2):
            raise ValueError("num_cells_range must be a tuple of two integers.")

        if num_empty_cells is None:
            # Loop over the specified range of missing cells and generate puzzles accordingly
            for i in range(num_cells_range[0], num_cells_range[1] + 1):
                for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {i} missing cells"):
                    sudoku = Sudoku(grid_size=self.size, num_missing_cells=i)
                    self._append_quantum_resources(sudoku)
        else:
            # Generate puzzles with a specific number of missing cells
            for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {num_empty_cells} missing cells"):
                sudoku = Sudoku(grid_size=self.size, num_missing_cells=num_empty_cells)
                self._append_quantum_resources(sudoku)

        # Return the collected data as a DataFrame
        return pd.DataFrame(self.data_list)

    def insert_into_sql(self) -> None:
        """
        Insert the computed quantum resource data into the 'quantum_resources' SQL table.

        This method uses the SQLAlchemy engine (if provided) and:
            - Creates the table if it does not exist.
            - Converts the data list into a DataFrame.
            - Inserts the DataFrame into the SQL table with the appropriate data types.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            # Ensure the table exists before inserting data
            self._create_table_if_not_exists()
            df = pd.DataFrame(self.data_list)
            dtype = {
                'missing_cells': Integer(),
                'num_qubits_simple_encoding': Integer(),
                'mcx_gates_simple_encoding': Integer(),
                'total_gates_simple_encoding': Integer(),
                'num_qubits_pattern_encoding': Integer(),
                'mcx_gates_pattern_encoding': Integer(),
                'total_gates_pattern_encoding': Integer(),
            }
            # Insert data into the SQL table using multi-row insertion for efficiency
            df.to_sql('quantum_resources', self.engine, if_exists='append', index=False, dtype=dtype, method='multi')
        except Exception as e:
            logger.error(f"Error inserting data into SQL: {e}")
            raise

    def load_db_table_to_df(self, table_name: str) -> pd.DataFrame:
        """
        Load data from a specified database table into a pandas DataFrame.

        Parameters:
            table_name: The name of the table to load data from.

        Returns:
            A pandas DataFrame containing the data from the specified table.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}")
            raise

    def _append_quantum_resources(self, sudoku: Sudoku) -> None:
        """
        Compute and append quantum resource metrics for a single Sudoku puzzle to the data list.

        Parameters:
            sudoku: A Sudoku instance representing the puzzle.
        """
        # Retrieve quantum resources for both simple and pattern encodings
        sim_num_qubits, sim_mcx_gates, sim_total_gates,  \
        pat_num_qubits, pat_mcx_gates, pat_total_gates  = self.find_quantum_resources(sudoku)
        
        # Append the data as a dictionary to the internal list
        self.data_list.append({
            'missing_cells': sudoku.num_missing_cells,
            'num_qubits_simple_encoding': sim_num_qubits,
            'mcx_gates_simple_encoding': sim_mcx_gates,
            'total_gates_simple_encoding': sim_total_gates,
            'num_qubits_pattern_encoding': pat_num_qubits,
            'mcx_gates_pattern_encoding': pat_mcx_gates,
            'total_gates_pattern_encoding': pat_total_gates,
        })

    def _create_table_if_not_exists(self) -> None:
        """
        Create the 'quantum_resources' table in the database if it doesn't exist.

        This method uses SQLAlchemy's metadata to create all tables defined by the Base class.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        Base.metadata.create_all(self.engine)

# ------------------------------------------------------------
# Data Analysis and Visualization Class
# ------------------------------------------------------------
class QuantumDataAnalysis:
    """
    Class for analyzing and visualizing the quantum resource metrics collected from Sudoku puzzles.

    Attributes:
        df: A pandas DataFrame containing the quantum resource data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_statistics(self) -> pd.DataFrame:
        """
        Compute and return descriptive statistics for each feature in the DataFrame.

        Returns:
            A pandas DataFrame with summary statistics (mean, std, quartiles, etc.) for each column.
        """
        return self.df.describe()

    def plot_distributions_pdf(self, file_path=None) -> None:
        """
        Plot histograms for key quantum resource metrics and save all plots into a single PDF file.

        This method generates:
            - Histograms for each resource metric (with a KDE overlay).
            - A correlation heatmap.
            - A pairplot to visualize relationships between features.

        The plots are saved into 'quantum_resource_distributions.pdf' (or a custom file if provided).

        Parameters:
            file_path: Optional; custom file path for the PDF report.
        """
        pdf_filename = file_path or 'quantum_resource_distributions.pdf'
        metrics = [
            'num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding',
            'num_qubits_pattern_encoding', 'mcx_gates_pattern_encoding', 'total_gates_pattern_encoding'
        ]
        
        # Define plot settings for consistency
        bins = 20         # Number of bins for histograms
        dpi = 80          # Resolution for saved plots
        figure_size = (8, 6)  # Figure size for each plot
        
        with PdfPages(pdf_filename) as pdf:
            # Generate histogram plots for each metric
            for metric in metrics:
                try:
                    plt.figure(figsize=figure_size)
                    sns.histplot(data=self.df, x=metric, kde=True, bins=bins)
                    plt.title(f'Distribution of {metric}')
                    pdf.savefig(dpi=dpi)
                    plt.close()
                except Exception as e:
                    logger.error(f"Error plotting {metric}: {e}")
                    plt.close()

            try:
                # Generate a correlation heatmap
                plt.figure(figsize=figure_size)
                correlation = self.df.corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Heatmap')
                pdf.savefig(dpi=dpi)
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting correlation heatmap: {e}")
                plt.close()

            try:
                # Generate a pairplot; downsample if the dataset is large for performance reasons
                small_df = self.df.sample(frac=0.1) if len(self.df) > 1000 else self.df
                sns.pairplot(small_df)
                pdf.savefig(dpi=dpi)
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting pair plots: {e}")
                plt.close()

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute and return the correlation matrix for the DataFrame features.

        Returns:
            A pandas DataFrame representing the correlation coefficients between features.
        """
        return self.df.corr()

    def plot_distributions(self) -> None:
        """
        Plot and save histograms for key quantum resource metrics as separate PNG image files.
        """
        metrics = [
            'num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding', 
            'num_qubits_pattern_encoding', 'mcx_gates_pattern_encoding', 'total_gates_pattern_encoding'
        ]
        for metric in metrics:
            sns.histplot(data=self.df, x=metric, kde=True)
            plt.title(f'Distribution of {metric}')
            plt.savefig(f'{metric}_distribution.png')
            plt.close()

    def pair_plots(self) -> None:
        """
        Generate pair plots for the resource metrics grouped by encoding type and save as PNG images.

        This method creates:
            - A pair plot for simple encoding metrics.
            - A pair plot for pattern encoding metrics.
        """
        simple_encoding_metrics = [
            'num_qubits_simple_encoding', 
            'mcx_gates_simple_encoding',
            'total_gates_simple_encoding', 
        ]
        
        pattern_encoding_metrics = [
            'num_qubits_pattern_encoding', 
            'mcx_gates_pattern_encoding',
            'total_gates_pattern_encoding', 
        ]
        
        # Plot pair plot for simple encoding metrics
        g1 = sns.pairplot(self.df[simple_encoding_metrics])
        g1.fig.suptitle('Simple Encoding Metrics Pair Plot', y=1.02)
        plt.savefig('pair_plots_simple_encoding.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot pair plot for pattern encoding metrics
        g2 = sns.pairplot(self.df[pattern_encoding_metrics])
        g2.fig.suptitle('Pattern Encoding Metrics Pair Plot', y=1.02)
        plt.savefig('pair_plots_pattern_encoding.png', dpi=300, bbox_inches='tight')
        plt.close()

    def full_correlation_heatmap(self, file_name='full_correlation_heatmap.png') -> None:
        """
        Plot and save a heatmap showing the full correlation matrix between all features.

        Parameters:
            file_name: The name of the file to save the heatmap image.
        """
        correlation = self.df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.savefig(file_name, dpi=80)
        plt.close()

    def plot_correlation_heatmaps(self) -> None:
        """
        Plot and save separate heatmaps for simple encoding and pattern encoding metrics.
        """
        simple_encoding_metrics = [
            'num_qubits_simple_encoding', 
            'mcx_gates_simple_encoding',
            'total_gates_simple_encoding'
        ]
        
        pattern_encoding_metrics = [
            'num_qubits_pattern_encoding',
            'mcx_gates_pattern_encoding',
            'total_gates_pattern_encoding' 
        ]
        
        # Heatmap for simple encoding metrics
        correlation_simple = self.df[simple_encoding_metrics].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_simple, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap - Simple Encoding')
        plt.savefig('correlation_heatmap_simple_encoding.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap for pattern encoding metrics
        correlation_pattern = self.df[pattern_encoding_metrics].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_pattern, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap - Pattern Encoding')
        plt.savefig('correlation_heatmap_pattern_encoding.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_interactive_correlation_heatmap(self) -> None:
        """
        Generate and display an interactive correlation heatmap using Plotly.

        The interactive heatmap allows for hover information and zooming.
        """
        correlation = self.df.corr()
        fig = px.imshow(correlation, text_auto=True, title='Interactive Correlation Heatmap of Quantum Resources')
        fig.show()

    def perform_t_tests(self) -> None:
        """
        Perform independent two-sample t-tests comparing quantum resource metrics between
        simple and pattern encodings, and log the results.

        For each key metric (number of qubits, MCX gates, total gates), the method:
            - Finds the corresponding metric for the other encoding.
            - Performs a t-test assuming unequal variances.
            - Logs the t-statistic and p-value.
        """
        from scipy.stats import ttest_ind
        metrics = ['num_qubits_simple_encoding', 'mcx_gates_simple_encoding', 'total_gates_simple_encoding']
        for column in metrics:
            pattern_column = column.replace('simple', 'pattern')
            t_stat, p_value = ttest_ind(self.df[column], self.df[pattern_column], equal_var=False, nan_policy='omit')
            logger.info(f"T-test for {column} vs {pattern_column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# ------------------------------------------------------------
# Example Usage (Commented Out)
# ------------------------------------------------------------
# The following lines show how you might use the GenData and QuantumDataAnalysis classes.
# Uncomment and modify as needed:
#
# gen_data = GenData(grid_size=2, num_puzzles=10)
# data_df = gen_data.generate_data(num_empty_cells=5)
# analysis = QuantumDataAnalysis(df=data_df)
# stats = analysis.get_statistics()
# analysis.plot_distributions_pdf()
