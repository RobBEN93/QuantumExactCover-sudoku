
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

warnings.filterwarnings('ignore')

from python_package.Sudoku import Sudoku
from python_package.quantum import ExactCoverQuantumSolver

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy base class for model definition
Base = declarative_base()

# SQLAlchemy model for representing the 'quantum_resources' table in the database
class QuantumResources(Base):
    __tablename__ = 'quantum_resources'

    id = Column(Integer, primary_key=True)
    missing_cells = Column(Integer)
    num_qubits_sim = Column(Integer)
    total_gates_simple_encoding = Column(Integer)
    mcx_gates_simple_encoding = Column(Integer)
    num_qubits_pat = Column(Integer)
    total_gates_pattern_encoding = Column(Integer)
    mcx_gates_pattern_encoding = Column(Integer)

class GenData:
    """
    Class for generating Sudoku puzzles and estimating the quantum resources required for solving them.
    """
    def __init__(self, grid_size: int = 2, num_puzzles: int = 1000, db_url: Optional[str] = None):
        # Initialize grid size, number of puzzles, database URL, and other relevant attributes
        self.size = grid_size
        self.board_size = grid_size * grid_size
        self.num_puzzles = num_puzzles
        self.db_url = db_url
        self.total_cells = self.board_size * self.board_size
        self.data_list: List[Dict[str, Any]] = []
        # Create a database engine if a URL is provided
        if db_url:
            self.engine = create_engine(self.db_url)
        else:
            self.engine = None


    def load_db_table_to_df(self, table_name: str) -> pd.DataFrame:
        """
        Load data from a database table into a DataFrame.
        
        Parameters:
        - table_name: The name of the table to load data from.
        
        Returns:
        - A DataFrame containing the data from the specified table.
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
        Append quantum resource data for the given Sudoku puzzle to the data list.
        
        Parameters:
        - sudoku: A Sudoku instance representing the puzzle to be solved.
        """
        # Compute quantum resources for the given Sudoku puzzle using both encodings
        sim_num_qubits, sim_total_gates, sim_mcx_gates, \
        pat_num_qubits, pat_total_gates, pat_mcx_gates = self.find_quantum_resources(sudoku)
        
        # Append the computed data to the data list
        self.data_list.append({
            'missing_cells': sudoku.num_missing_cells,
            'num_qubits_sim': sim_num_qubits,
            'total_gates_simple_encoding': sim_total_gates,
            'mcx_gates_simple_encoding': sim_mcx_gates,
            'num_qubits_pat': pat_num_qubits,
            'total_gates_pattern_encoding': pat_total_gates,
            'mcx_gates_pattern_encoding': pat_mcx_gates
        })

    def _create_table_if_not_exists(self) -> None:
        """
        Create the 'quantum_resources' table in the database if it does not already exist.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        # Create all tables defined by the Base class
        Base.metadata.create_all(self.engine)

    def insert_into_sql(self) -> None:
        """
        Insert the data into the 'quantum_resources' table in the database.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            # Ensure the table exists
            self._create_table_if_not_exists()
            # Convert data list to DataFrame
            df = pd.DataFrame(self.data_list)
            # Define data types for the DataFrame columns
            dtype = {
                'missing_cells': Integer(),
                'num_qubits_sim': Integer(),
                'total_gates_simple_encoding': Integer(),
                'mcx_gates_simple_encoding': Integer(),
                'num_qubits_pat': Integer(),
                'total_gates_pattern_encoding': Integer(),
                'mcx_gates_pattern_encoding': Integer(),
            }
            # Insert data into the SQL table
            df.to_sql('quantum_resources', self.engine, if_exists='append', index=False, dtype=dtype, method='multi')
        except Exception as e:
            logger.error(f"Error inserting data into SQL: {e}")
            raise

    def generate_data(self, num_empty_cells: int = None, num_cells_range: tuple = (1, 9)) -> pd.DataFrame:
        """
        Generate Sudoku puzzles and compute quantum resources for each puzzle.
        
        Parameters:
        - num_empty_cells: Specific number of missing cells to generate puzzles with (optional).
        - num_cells_range: Tuple specifying a range of missing cells if num_empty_cells is not provided.
        
        Returns:
        - A DataFrame containing the quantum resource data for the generated puzzles.
        """
        if num_empty_cells is not None:
            if not isinstance(num_empty_cells, int) or num_empty_cells < 0:
                raise ValueError("num_empty_cells must be a non-negative integer.")
        if not (isinstance(num_cells_range, tuple) and len(num_cells_range) == 2):
            raise ValueError("num_cells_range must be a tuple of two integers.")

        if num_empty_cells is None:
            # Loop over a range of missing cells if num_empty_cells is not specified
            for i in range(num_cells_range[0], num_cells_range[1] + 1):
                for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {i} missing cells"):
                    sudoku = Sudoku(grid_size=self.size, num_missing_cells=i)
                    self._append_quantum_resources(sudoku)
        else:
            # Generate data for a specific number of missing cells
            for _ in tqdm(range(self.num_puzzles), desc=f"Generating puzzles with {num_empty_cells} missing cells"):
                sudoku = Sudoku(grid_size=self.size, num_missing_cells=num_empty_cells)
                self._append_quantum_resources(sudoku)

        return pd.DataFrame(self.data_list)

    def find_quantum_resources(self, sudoku: Sudoku):
        """
        Compute quantum resources required for solving a Sudoku puzzle using simple and pattern encodings.
        
        Parameters:
        - sudoku: A Sudoku instance representing the puzzle to be solved.
        
        Returns:
        - A tuple containing quantum resource data for both encodings (simple and pattern).
        """
        # Compute quantum resources for simple encoding
        circ = ExactCoverQuantumSolver(sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_total_gates, sim_mcx_gates = circ.find_resources()
        # Compute quantum resources for pattern encoding
        circ = ExactCoverQuantumSolver(sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_total_gates, pat_mcx_gates = circ.find_resources()
        return sim_num_qubits, sim_total_gates, sim_mcx_gates, pat_num_qubits, pat_total_gates, pat_mcx_gates

class QuantumDataAnalysis:
    """
    Class for analyzing and visualizing quantum resource metrics.
    """
    def __init__(self, df: pd.DataFrame):
        # Initialize with a DataFrame containing quantum resource data
        self.df = df

    def get_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics of the DataFrame.
        
        Returns:
        - A DataFrame containing descriptive statistics for each column.
        """
        return self.df.describe()

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute the correlation matrix of the DataFrame.
        
        Returns:
        - A DataFrame representing the correlation coefficients between features.
        """
        return self.df.corr()

    def plot_distributions(self) -> None:
        """
        Plot distributions of key quantum resource metrics and save them as image files.
        """
        metrics = [
            'num_qubits_sim', 'total_gates_simple_encoding', 'mcx_gates_simple_encoding',
            'num_qubits_pat', 'total_gates_pattern_encoding', 'mcx_gates_pattern_encoding'
        ]
        for metric in metrics:
            # Plot the distribution of each metric
            sns.histplot(data=self.df, x=metric, kde=True)
            plt.title(f'Distribution of {metric}')
            plt.savefig(f'{metric}_distribution.png')
            plt.close()

    def pair_plots(self) -> None:
        """
        Generate pair plots to visualize relationships between key metrics and save the plot.
        """
        # Generate pair plots to visualize correlations between different metrics
        sns.pairplot(self.df)
        plt.savefig('pair_plots.png')
        plt.close()

    def plot_distributions_pdf(self) -> None:
        """
        Plot distributions of key quantum resource metrics and save them all into a single PDF report.
        """
        pdf_filename = 'quantum_resource_distributions.pdf'
        with PdfPages(pdf_filename) as pdf:
            metrics = [
                'num_qubits_sim', 'total_gates_simple_encoding', 'mcx_gates_simple_encoding',
                'num_qubits_pat', 'total_gates_pattern_encoding', 'mcx_gates_pattern_encoding'
            ]
            for metric in metrics:
                # Plot the distribution of each metric
                sns.histplot(data=self.df, x=metric, kde=True)
                plt.title(f'Distribution of {metric}')
                pdf.savefig()  # Save the current figure to the PDF
                plt.close()
                
            correlation = self.df.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
            pdf.savefig()
            
            sns.pairplot(self.df)
            pdf.savefig()
            
    def plot_interactive_correlation_heatmap(self) -> None:
        """
        Plot an interactive heatmap showing correlations between different features.
        """
        correlation = self.df.corr()
        fig = px.imshow(correlation, 
                        text_auto=True, 
                        title='Interactive Correlation Heatmap of Quantum Resources')
        fig.show()

    def plot_correlation_heatmap(self) -> None:
        """
        Plot a heatmap showing correlations between different features and save it as an image file.
        """
        # Compute the correlation matrix and plot it as a heatmap
        correlation = self.df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Quantum Resources')
        plt.savefig('correlation_heatmap.png')
        plt.close()

    def perform_t_tests(self) -> None:
        """
        Perform t-tests to compare the quantum resources between different encodings (simple vs pattern).
        """
        from scipy.stats import ttest_ind
        # Loop through key metrics and perform t-tests between simple and pattern encodings
        for column in ['num_qubits_sim', 'total_gates_simple_encoding', 'mcx_gates_simple_encoding']:
            pattern_column = column.replace('simple', 'pattern')
            t_stat, p_value = ttest_ind(self.df[column], self.df[pattern_column], equal_var=False, nan_policy='omit')
            logger.info(f"T-test for {column} vs {pattern_column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# Example use of QuantumDataAnalysis class
# gen_data = GenData(grid_size=2, num_puzzles=10)
# data_df = gen_data.generate_data(num_empty_cells=5)
# analysis = QuantumDataAnalysis(df=data_df)