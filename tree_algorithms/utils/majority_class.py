import pandas as pd
from collections import Counter

def majority_class(dataset: pd.DataFrame, target_column: str) -> str:
    """
    Return the most frequent class in the target column.

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - target_column (str): The name of the target class column.

    Returns:
    - str: The majority class.
    """
    counts = Counter(dataset[target_column])
    return counts.most_common(1)[0][0]
