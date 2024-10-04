import pandas as pd
from sqlalchemy import create_engine, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

from python_package.Sudoku import Sudoku
from python_package.exact_cover_circ import ExactCoverQuantumSolver

Base = declarative_base()

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
        sns.scatterplot(data=self.df, x=x_col, y=y_col)
        plt.show()
        
    def get_statistics(self) -> pd.DataFrame:
        return self.df.describe()
    
    def load_db_table_to_df(self, table_name: str) -> pd.DataFrame:
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
        if stats_dict is None:
            return self.df.describe(include='all')
        else:
            return self.df.agg(stats_dict)
    
    def correlation_matrix(self) -> pd.DataFrame:
        return self.df.corr()
    
    def create_table_if_not_exists(self) -> None:
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        Base.metadata.create_all(self.engine)
    
    def insert_into_sql(self) -> None:
        if self.engine is None:
            raise ValueError("Database engine is not set. Please provide a valid db_url when initializing GenData.")
        try:
            # Ensure the table exists
            self.create_table_if_not_exists()
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
        circ = ExactCoverQuantumSolver(sudoku, simple=True, pattern=False)
        sim_num_qubits, sim_total_gates, sim_mcx_gates = circ.find_resources()
        circ = ExactCoverQuantumSolver(sudoku, simple=False, pattern=True)
        pat_num_qubits, pat_total_gates, pat_mcx_gates = circ.find_resources()
        return sim_num_qubits, sim_total_gates, sim_mcx_gates, pat_num_qubits, pat_total_gates, pat_mcx_gates