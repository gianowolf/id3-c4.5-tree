import pandas as pd
from typing import List, Literal, Optional, Tuple
from tree_algorithms.utils.information_gain import information_gain
from tree_algorithms.utils.gain_ratio import gain_ratio
from tree_algorithms.utils.is_numeric import is_numeric_attribute
from tree_algorithms.utils.best_numeric_split import best_numeric_split

def best_split_attribute(
    dataset: pd.DataFrame,
    attributes: List[str],
    target_column: str,
    method: Literal["gain", "gain_ratio"] = "gain"
) -> Optional[Tuple[str, Optional[float]]]:
    """
    Determine the best attribute to split on using a specified method.
    For numeric attributes, also find the best threshold.

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attributes (List[str]): List of attribute names to consider.
    - target_column (str): The class label column.
    - method (str): "gain" for ID3 or "gain_ratio" for C4.5.

    Returns:
    - Optional[Tuple[str, Optional[float]]]: (attribute, threshold) or None
    """
    best_attr = None
    best_score = -float("inf")
    best_threshold = None

    for attr in attributes:
        series = dataset[attr]
        if is_numeric_attribute(series) and method == "gain_ratio":
            threshold, score = best_numeric_split(dataset, attr, target_column)
        else:
            threshold = None
            score = (
                information_gain(dataset, attr, target_column)
                if method == "gain"
                else gain_ratio(dataset, attr, target_column)
            )

        if score > best_score:
            best_score = score
            best_attr = attr
            best_threshold = threshold

    if best_attr is None:
        return None

    return best_attr, best_threshold
