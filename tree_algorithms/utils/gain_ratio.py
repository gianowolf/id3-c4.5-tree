from tree_algorithms.utils.information_gain import information_gain
from tree_algorithms.utils.intrinsic_value import intrinsic_value
import pandas as pd

def gain_ratio(dataset: pd.DataFrame, attribute: str, target_column: str) -> float:
    """
    Compute the Gain Ratio of an attribute.

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attribute (str): The attribute to evaluate.
    - target_column (str): The name of the class label column.

    Returns:
    - float: The Gain Ratio.
    """
    gain = information_gain(dataset, attribute, target_column)
    iv = intrinsic_value(dataset, attribute)

    if iv == 0:
        return 0.0

    return gain / iv
