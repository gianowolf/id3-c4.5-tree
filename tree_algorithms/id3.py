import pandas as pd
import numpy as np
from tree_algorithms.utils.best_split import best_split_attribute
from tree_algorithms.utils.split_dataset import split_dataset
from tree_algorithms.utils.majority_class import majority_class

def build_tree(dataset, attributes, target_column, min_gain=0.0, method="gain", allow_numeric=False, max_depth=None, depth=0):
    if not allow_numeric:
        attributes = [
            attr for attr in attributes
            if not np.issubdtype(dataset[attr].dtype, np.number)
        ]

    if len(dataset[target_column].unique()) == 1:
        return dataset[target_column].iloc[0]

    if not attributes or (max_depth is not None and depth >= max_depth):
        return majority_class(dataset, target_column)

    result = best_split_attribute(dataset, attributes, target_column, method=method)
    if result is None:
        return majority_class(dataset, target_column)

    best_attr, threshold, gain_value = result
    if gain_value < min_gain:
        return majority_class(dataset, target_column)

    if not allow_numeric and threshold is not None:
        return majority_class(dataset, target_column)

    tree = {best_attr: {}}
    attr_key = (best_attr, threshold) if threshold is not None else best_attr
    subsets = split_dataset(dataset, attr_key)

    for attr_val, subset in subsets.items():
        if subset.empty:
            tree[best_attr][attr_val] = majority_class(dataset, target_column)
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][attr_val] = build_tree(
                subset,
                remaining_attrs,
                target_column,
                min_gain=min_gain,
                method=method,
                allow_numeric=allow_numeric,
                max_depth=max_depth,
                depth=depth + 1
            )

    return tree

def predict(tree, sample):
    """
    Realiza una predicción para una muestra dada usando un árbol ID3.

    Parameters:
    - tree (dict or str): Árbol entrenado o clase hoja.
    - sample (dict): Muestra a clasificar.

    Returns:
    - str: Clase predicha.
    """
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    if value not in tree[attribute]:
        return None

    return predict(tree[attribute][value], sample)
