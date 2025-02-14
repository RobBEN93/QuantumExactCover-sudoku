"""
This script provides classes and functions for generating, analyzing, and storing data related to quantum resources required to solve Sudoku puzzles.

Classes:
- QuantumResources: SQLAlchemy model representing the structure of the 'quantum_resources' table for database storage.
- GenData: Main class for generating sudoku puzzles using python_package.Sudoku, estimating the quantum resources needed 
to solve them, and handling the data.

Dependencies:
- pandas: Used for data manipulation and analysis.
- SQLAlchemy: Used for database management.
- matplotlib, seaborn: Used for data visualization.
- python_package (Sudoku, ExactCoverQuantumSolver): Used for puzzle generation and quantum problem-solving.

"""

import pandas as pd
from sqlalchemy import create_engine, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from sudoku_nisq.sudoku import Sudoku
from sudoku_nisq.quantum import ExactCoverQuantumSolver

# SQLAlchemy base class for model definition
Base = declarative_base()

""" Min number of cells for a 9x9 sudoku to have a single solution"""

class QuantumResources(Base):
    """
    SQLAlchemy model for representing the 'quantum_resources' table in the database.
    """
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

    Attributes:
    - grid_size: The size of the Sudoku grid (e.g., 2 for a 4x4 grid).
    - num_puzzles: Number of puzzles to generate.
    - db_url: URL for the SQL database (optional).
    - engine: SQLAlchemy engine for database interactions.
    - df: DataFrame for storing quantum resource information.
    """
    def __init__(self, grid_size: int = 2, num_puzzles: int = 1000, db_url: Optional[str] = None):
        self.size = grid_size
        self.board_size = grid_size * grid_size
        self.num_puzzles = num_puzzles
        self.db_url = db_url
        self.total_cells = self.board_size * self.board_size
        self.df = pd.DataFrame(columns=[
            'missing_cells', 'num_qubits_sim', 'total_gates_simple_encoding',
            'mcx_gates_simple_encoding', 'num_qubits_pat', 'total_gates_pattern_encoding',
            'mcx_gates_pattern_encoding'
        ])
        if db_url:
            self.engine = create_engine(self.db_url)
        else:
            self.engine = None

    def _quantum_dataframe(self, sudoku: Sudoku) -> None:
        """
        Generate a DataFrame row for the quantum resources required to solve a given Sudoku puzzle.
        
        Args:
        - sudoku: A Sudoku instance representing the puzzle to be solved.
        """
        sim_num_qubits, sim_total_gates, sim_mcx_gates, \
        pat_num_qubits, pat_total_gates, pat_mcx_gates = self.find_quantum_resources(sudoku)
        new_data = pd.DataFrame({
            'missing_cells': [sudoku.num_missing_cells],
            'num_qubits_sim': [sim_num_qubits],
            'total_gates_simple_encoding': [sim_total_gates],
            'mcx_gates_simple_encoding': [sim_mcx_gates],
            'num_qubits_pat': [pat_num_qubits],
            'total_gates_pattern_encoding': [pat_total_gates],
            'mcx_gates_pattern_encoding': [pat_mcx_gates]
        })
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        
    def plot_data(self, x_col: str = 'num_qubits_sim', y_col: str = 'mcx_gates_simple_encoding') -> None:
        """
        Plot the relationship between two columns from the DataFrame.
        
        Args:
        - x_col: Column name for x-axis.
        - y_col: Column name for y-axis.
        """
        sns.scatterplot(data=self.df, x=x_col, y=y_col)
        plt.show()
        
    def get_statistics(self) -> pd.DataFrame:
        """
        Get descriptive statistics for the DataFrame.
        
        Returns:
        - A DataFrame containing statistics for each column.
        """
        return self.df.describe()
    
    def load_db_table_to_df(self, table_name: str) -> pd.DataFrame:
        """
        Load data from a database table into a DataFrame.
        
        Args:
        - table_name: The name of the table to load data from.
        
        Returns:
        - A DataFrame containing the data from the specified table.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Error: {e}")
            df = pd.DataFrame()
        return df
    
    def custom_stats(self, stats_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compute custom statistics for the DataFrame.
        
        Args:
        - stats_dict: Dictionary specifying the statistics to compute for each column.
        
        Returns:
        - A DataFrame containing the specified statistics.
        """
        if stats_dict is None:
            return self.df.describe(include='all')
        else:
            return self.df.agg(stats_dict)
    
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute the correlation matrix for the DataFrame.
        
        Returns:
        - A DataFrame containing the correlation coefficients.
        """
        return self.df.corr()
    
    def _create_table_if_not_exists(self) -> None:
        """
        Create the 'quantum_resources' table in the database if it does not already exist.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        Base.metadata.create_all(self.engine)
    
    def insert_into_sql(self) -> None:
        """
        Insert the DataFrame data into the 'quantum_resources' table in the database.
        """
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            # Ensure the table exists
            self._create_table_if_not_exists()
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
            self.df.to_sql('quantum_resources', self.engine, if_exists='append', index=False, dtype=dtype)
        except Exception as e:
            print(f"Error: {e}")
        
    def generate_data(self, num_empty_cells: int = None, num_cells_range: tuple = (1, 9)) -> pd.DataFrame:
        """
        Generate Sudoku puzzles and compute quantum resources for each puzzle.
        
        Args:
        - num_empty_cells: Specific number of missing cells (optional).
        - num_cells_range: Tuple specifying a range of missing cells (if num_empty_cells is not provided).
        
        Returns:
        - A DataFrame containing the quantum resource data for the generated puzzles.
        """
        if num_empty_cells is None:
            # Loop over a range of missing cells if num_empty_cells is not specified
            for i in range(num_cells_range[0], num_cells_range[1] + 1):
                for _ in range(self.num_puzzles):
                    sudoku = Sudoku(grid_size=self.size, num_missing_cells=i)
                    self._quantum_dataframe(sudoku)
        else:
            # Generate data for a specific number of missing cells
            for _ in range(self.num_puzzles):
                sudoku = Sudoku(grid_size=self.size, num_missing_cells=num_empty_cells)
                self._quantum_dataframe(sudoku)

        return self.df
        
    def find_quantum_resources(self, sudoku: Sudoku):
        """
        Compute quantum resources required for solving a Sudoku puzzle using simple and pattern encodings.
        
        Args:
        - sudoku: A Sudoku instance representing the puzzle to be solved.
        
        Returns:
        - Tuple containing quantum resource data for both encodings (simple and pattern).
        """
        circ = ExactCoverQuantumSolver(sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_total_gates, sim_mcx_gates = circ.find_resources()
        circ = ExactCoverQuantumSolver(sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_total_gates, pat_mcx_gates = circ.find_resources()
        return sim_num_qubits, sim_total_gates, sim_mcx_gates, pat_num_qubits, pat_total_gates, pat_mcx_gates
    
    
    def regression_analysis(self) -> None:
        """
        Perform regression analysis to understand the impact of the number of missing cells on quantum resources.
        For each key quantum metric, an Ordinary Least Squares (OLS) regression model is fitted.
        This helps in determining how missing cells influence the number of qubits or gates required.
        Regression analysis is used to model the relationship between a dependent variable (e.g., number of qubits)
        and one or more independent variables (e.g., number of missing cells). The resulting coefficients help
        quantify the effect of the independent variable on the dependent variable.
        """
        X = self.df[['missing_cells']]
        for col in ['num_qubits_sim', 'total_gates_simple_encoding', 'mcx_gates_simple_encoding']:
            y = self.df[col]
            X_const = sm.add_constant(X)  # Add constant for intercept
            model = sm.OLS(y, X_const).fit()
            print(f"Regression analysis for {col}:")
            print(model.summary())

    def model_resources_prediction(self) -> None:
        """
        Train a simple linear regression model to predict quantum resources based on the number of missing cells.
        This involves splitting the data into training and testing sets, fitting a Linear Regression model, 
        and evaluating its performance using Mean Squared Error (MSE).
        This helps in predicting quantum resource requirements and assessing the model's predictive capability.
        """
        features = ['missing_cells']
        target = 'num_qubits_sim'

        X = self.df[features]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        print(f'Mean Squared Error for predicting {target}: {mse:.4f}')
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted {target}')
        plt.savefig(f'actual_vs_predicted_{target}.png')
        plt.close()

    def feature_engineering(self) -> None:
        """
        Add custom features to enhance the analysis.
        New features include:
        - gates_per_qubit_simple: Ratio of total gates to number of qubits for simple encoding.
        - gates_per_qubit_pattern: Ratio of total gates to number of qubits for pattern encoding.
        
        Feature engineering is a critical step in data analysis and machine learning that involves creating
        new features from existing ones to make models more effective. By adding these features, we can
        improve the quality of insights gained from the data and potentially enhance the predictive performance
        of models.
        """
        self.df['gates_per_qubit_simple'] = self.df.apply(
            lambda row: row['total_gates_simple_encoding'] / row['num_qubits_sim']
            if row['num_qubits_sim'] != 0 else np.nan, axis=1)
        self.df['gates_per_qubit_pattern'] = self.df.apply(
            lambda row: row['total_gates_pattern_encoding'] / row['num_qubits_pat']
            if row['num_qubits_pat'] != 0 else np.nan, axis=1)

    def generate_summary_report(self) -> None:
        """
        Generate a comprehensive summary report of the analysis.
        The report includes descriptive statistics, correlation matrix, visualizations, t-tests, 
        regression analysis, and a predictive model evaluation.
        """
        print("Descriptive Statistics:\n", self.get_statistics())
        print("\nCorrelation Matrix:\n", self.correlation_matrix())

        self.plot_distributions()
        self.pair_plots()
        self.plot_correlation_heatmap()
        self.perform_t_tests()
        self.regression_analysis()
        self.model_resources_prediction()
