import pandas as pd
from typing import Dict, Optional, Union

def split_dataset(
    dataset: pd.DataFrame,
    attribute: str,
    threshold: Optional[Union[int, float]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset by attribute. If threshold is provided, split as numeric.

    Parameters:
    - dataset (pd.DataFrame): Dataset to split.
    - attribute (str): Attribute to split on.
    - threshold (float or int, optional): If set, perform binary numeric split.

    Returns:
    - dict: Dictionary of subsets keyed by attribute value or threshold condition.
    """
    if threshold is not None:
        lower = dataset[dataset[attribute] <= threshold]
        upper = dataset[dataset[attribute] > threshold]
        return {
            f"<= {threshold}": lower,
            f"> {threshold}": upper
        }
    else:
        return {value: subset for value, subset in dataset.groupby(attribute)}
