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
    Load dataset from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        numpy array with shape (n_samples, 2) containing x and y coordinates

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't have required x, y columns
    """
    df = pd.read_csv(file_path)

    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError(f"CSV file must contain 'x' and 'y' columns. Found: {df.columns.tolist()}")

    return df[['x', 'y']].to_numpy()
