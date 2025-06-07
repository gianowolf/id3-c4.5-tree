## Gain(A)=H(target)−H(target∣A)
from tree_algorithms.utils.entropy import entropy
from tree_algorithms.utils.attribute_entropy import attribute_entropy
import pandas as pd

def information_gain(dataset: pd.DataFrame, attribute: str, target_column: str) -> float:
    """
    Compute the information gain of splitting the dataset on a given attribute.
    
    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attribute (str): The attribute to evaluate.
    - target_column (str): The name of the target column.
    
    Returns:
    - float: The information gain value.
    """
    base_entropy = entropy(dataset, target_column)
    cond_entropy = attribute_entropy(dataset, attribute, target_column)
    return base_entropy - cond_entropy
