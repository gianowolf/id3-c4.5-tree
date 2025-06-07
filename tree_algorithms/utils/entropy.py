import math
from collections import Counter
import pandas as pd

def entropy(dataset: pd.DataFrame, target_column: str) -> float:
    """
    Compute the entropy of the target attribute in the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the target column.
    - target_column (str): The name of the column to compute entropy on.

    Returns:
    - float: Entropy value.
    """
    values = dataset[target_column]
    total = len(values)
    counts = Counter(values)

    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p) if p > 0 else 0

    return ent