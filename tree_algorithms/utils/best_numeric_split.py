import pandas as pd
import numpy as np
from tree_algorithms.utils.entropy import entropy

def best_numeric_split(dataset: pd.DataFrame, attribute: str, target_column: str):
    """
    Find the best threshold to split a numeric attribute using gain ratio.

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - attribute (str): Numeric attribute name.
    - target_column (str): Target class column.

    Returns:
    - (best_threshold: float, best_gain_ratio: float)
    """
    dataset = dataset.sort_values(attribute).reset_index(drop=True)
    values = dataset[attribute].values
    labels = dataset[target_column].values

    total_entropy = entropy(dataset, target_column)
    best_threshold = None
    best_gain_ratio = -float("inf")

    for i in range(1, len(values)):
        if labels[i] != labels[i - 1]:  # threshold only where class changes
            threshold = (values[i] + values[i - 1]) / 2

            left = dataset[dataset[attribute] <= threshold]
            right = dataset[dataset[attribute] > threshold]

            # Conditional entropy
            left_weight = len(left) / len(dataset)
            right_weight = len(right) / len(dataset)
            cond_entropy = left_weight * entropy(left, target_column) + right_weight * entropy(right, target_column)

            gain = total_entropy - cond_entropy

            # Intrinsic value (for gain ratio)
            iv = 0
            for weight in [left_weight, right_weight]:
                if weight > 0:
                    iv -= weight * np.log2(weight)

            gain_ratio = gain / iv if iv > 0 else 0

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

    return best_threshold, best_gain_ratio
