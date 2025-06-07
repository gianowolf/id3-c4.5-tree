import pandas as pd
from tree_algorithms.utils.discretize import discretize_numeric_features

def test_discretize_basic_case():
    df = pd.DataFrame({
        "Kills": [1, 2, 3, 4, 5],
        "Result": ["Win", "Lose", "Win", "Lose", "Win"]
    })
    result = discretize_numeric_features(df, bins=3)

    # Verifica que la columna fue reemplazada por valores categóricos
    assert result["Kills"].dtype.name == "category"
    assert all(str(val).startswith("Kills_bin_") for val in result["Kills"])

def test_discretize_multiple_numeric_columns():
    df = pd.DataFrame({
        "Kills": [1, 2, 3, 4, 5],
        "GPM": [300, 400, 500, 600, 700],
        "Result": ["Win"] * 5
    })
    result = discretize_numeric_features(df, bins=4)

    assert result["Kills"].dtype.name == "category"
    assert result["GPM"].dtype.name == "category"
    assert all("Kills_bin_" in str(v) for v in result["Kills"])
    assert all("GPM_bin_" in str(v) for v in result["GPM"])

def test_discretize_preserves_non_numeric():
    df = pd.DataFrame({
        "Hero": ["Invoker", "PA", "CM"],
        "Kills": [1, 5, 9]
    })
    result = discretize_numeric_features(df, bins=2)

    assert "Hero" in result.columns
    assert result["Hero"].equals(df["Hero"])

def test_discretize_correct_number_of_bins():
    df = pd.DataFrame({
        "Value": [10, 20, 30, 40, 50, 60]
    })
    result = discretize_numeric_features(df, bins=2)
    # Debe haber dos categorías posibles
    assert set(result["Value"].cat.categories) == {"Value_bin_1", "Value_bin_2"}
