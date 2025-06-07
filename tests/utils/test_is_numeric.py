import pandas as pd
from tree_algorithms.utils.is_numeric import is_numeric_attribute

def test_numeric_with_integers():
    series = pd.Series([1, 2, 3, 4])
    assert is_numeric_attribute(series) is True

def test_numeric_with_floats():
    series = pd.Series([1.0, 2.5, 3.3])
    assert is_numeric_attribute(series) is True

def test_non_numeric_with_strings():
    series = pd.Series(["Invoker", "PA", "CM"])
    assert is_numeric_attribute(series) is False

def test_non_numeric_with_booleans():
    series = pd.Series([True, False, True])
    # Se acepta que es numérico
    assert is_numeric_attribute(series) is True

def test_non_numeric_with_categorical():
    series = pd.Series(["a", "b", "c"], dtype="category")
    assert is_numeric_attribute(series) is False
