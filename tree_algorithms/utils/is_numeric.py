import pandas as pd

def is_numeric_attribute(series: pd.Series) -> bool:
    """
    Determine if a pandas Series represents a numeric attribute.

    Parameters:
    - series (pd.Series): The column to inspect.

    Returns:
    - bool: True if numeric (int or float), False otherwise.
    """
    return pd.api.types.is_numeric_dtype(series)
