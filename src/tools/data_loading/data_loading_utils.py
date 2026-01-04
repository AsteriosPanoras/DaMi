"""
Data loading utilities.

This module provides utility functions for loading and validating dataset files.
The synthetic data generation has been moved to scripts/generate_datasets.py,
and data loading has been simplified to use pre-generated datasets from the data/ folder.
"""

import pandas as pd
import numpy as np


def _load_data_from_file(file_path: str) -> np.ndarray:
    """
    Load dataset from a CSV or TXT file.

    Args:
        file_path: Path to the CSV or TXT file

    Returns:
        numpy array with shape (n_samples, 2) containing x and y coordinates

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't have required x, y columns or correct format
    """
    # Try loading with headers first
    try:
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', engine='python')

        # Check if it has x,y columns
        if 'x' in df.columns and 'y' in df.columns:
            return df[['x', 'y']].to_numpy()
    except:
        pass

    # Try loading without headers (assumes first column = x, second = y)
    try:
        df = pd.read_csv(file_path, sep=',', header=None, on_bad_lines='skip', engine='python')

        # Must have at least 2 columns
        if df.shape[1] < 2:
            raise ValueError(f"File must have at least 2 columns. Found: {df.shape[1]}")

        # Use first two columns as x, y
        return df.iloc[:, :2].to_numpy()
    except Exception as e:
        raise ValueError(f"Could not load file {file_path}. Error: {str(e)}")
