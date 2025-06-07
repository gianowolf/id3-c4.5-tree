import math
import pandas as pd

def intrinsic_value(dataset: pd.DataFrame, attribute: str) -> float:
    """
    Compute the intrinsic value of an attribute (used in Gain Ratio).

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attribute (str): The attribute to evaluate.

    Returns:
    - float: The intrinsic value (IV).
    """
    total = len(dataset)
    iv = 0.0

    for _, subset in dataset.groupby(attribute):
        p = len(subset) / total
        if p > 0:
            iv -= p * math.log2(p)

    return iv
