import pandas as pd
import pytest
from tree_algorithms.utils.entropy import entropy

def test_entropy_binary_distribution():
    df = pd.DataFrame({"Result": ["Win", "Lose"]})
    result = entropy(df, "Result")
    assert round(result, 3) == 1.000

def test_entropy_uniform_distribution():
    df = pd.DataFrame({"Result": ["Win", "Lose", "Win", "Lose"]})
    result = entropy(df, "Result")
    assert round(result, 3) == 1.000

def test_entropy_skewed_distribution():
    df = pd.DataFrame({"Result": ["Win", "Win", "Win", "Lose"]})
    result = entropy(df, "Result")
    assert round(result, 3) == 0.811  # Approximate

def test_entropy_pure_distribution():
    df = pd.DataFrame({"Result": ["Win", "Win", "Win", "Win"]})
    result = entropy(df, "Result")
    assert result == 0.0
