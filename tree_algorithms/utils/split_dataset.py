import pandas as pd
from typing import Dict, Optional, Union

def split_dataset(
    dataset: pd.DataFrame,
    attribute: Union[str, tuple[str, float]]
) -> Dict[str, pd.DataFrame]:
    """
    Divide el dataset según el atributo.
    Si es numérico, se espera una tupla (atributo, umbral).
    """
    if isinstance(attribute, tuple):
        attr_name, threshold = attribute
        return {
            f"<= {threshold}": dataset[dataset[attr_name] <= threshold],
            f"> {threshold}": dataset[dataset[attr_name] > threshold]
        }
    else:
        return {
            value: subset for value, subset in dataset.groupby(attribute, observed=False)
        }
