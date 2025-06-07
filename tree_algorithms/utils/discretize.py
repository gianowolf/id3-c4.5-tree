import pandas as pd

def discretize_numeric_features(dataset: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """
    Discretize all numeric features in the dataset into equal-width bins.

    Parameters:
    - dataset (pd.DataFrame): Input dataset.
    - bins (int): Number of categories to create per numeric attribute.

    Returns:
    - pd.DataFrame: Dataset with discretized numeric columns.
    """
    df = dataset.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        # Categoriza en bins con etiquetas ordinales
        df[col] = pd.cut(df[col], bins=bins, labels=[f"{col}_bin_{i+1}" for i in range(bins)])

    return df
