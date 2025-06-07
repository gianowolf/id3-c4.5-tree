import pandas as pd
from tree_algorithms.utils.entropy import entropy

def attribute_entropy(dataset: pd.DataFrame, attribute: str, target_column: str) -> float:
    """
    Compute the weighted entropy of a given attribute.
    
    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attribute (str): The attribute to split on.
    - target_column (str): The name of the class label column.
    
    Returns:
    - float: Weighted entropy after splitting by attribute.
    """
    total_instances = len(dataset)
    weighted_entropy = 0.0

    for value, subset in dataset.groupby(attribute):
        weight = len(subset) / total_instances
        weighted_entropy += weight * entropy(subset, target_column)

    return weighted_entropy
