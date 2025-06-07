import pandas as pd
from tree_algorithms.utils.majority_class import majority_class

def test_majority_class_simple():
    df = pd.DataFrame({"Result": ["Win", "Lose", "Lose", "Win", "Lose"]})
    result = majority_class(df, "Result")
    assert result == "Lose"

def test_majority_class_all_same():
    df = pd.DataFrame({"Result": ["Win"] * 5})
    result = majority_class(df, "Result")
    assert result == "Win"

def test_majority_class_tie():
    df = pd.DataFrame({"Result": ["Win", "Lose", "Lose", "Win"]})
    result = majority_class(df, "Result")
    # Puede devolver cualquiera de los dos — solo verificar que esté presente
    assert result in {"Win", "Lose"}

def test_majority_class_numeric_labels():
    df = pd.DataFrame({"Label": [1, 2, 2, 2, 1, 1, 2]})
    result = majority_class(df, "Label")
    assert result == 2
