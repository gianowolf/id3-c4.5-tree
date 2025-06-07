import pandas as pd
from tree_algorithms.utils.best_numeric_split import best_numeric_split

def test_best_numeric_split_simple_case():
    df = pd.DataFrame({
        "GPM": [300, 400, 500, 600],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    threshold, gr = best_numeric_split(df, "GPM", "Result")
    assert isinstance(threshold, float)
    assert gr > 0
    assert 400 < threshold < 500  # Debe estar entre Win → Lose

def test_best_numeric_split_no_class_change():
    df = pd.DataFrame({
        "GPM": [100, 200, 300, 400],
        "Result": ["Win", "Win", "Win", "Win"]
    })
    threshold, gr = best_numeric_split(df, "GPM", "Result")
    assert threshold is None
    assert gr == float("-inf") or gr < 0.0001  # no hay división válida

def test_best_numeric_split_multiple_changes():
    df = pd.DataFrame({
        "Kills": [1, 2, 3, 4, 5],
        "Result": ["Win", "Lose", "Win", "Lose", "Win"]
    })
    threshold, gr = best_numeric_split(df, "Kills", "Result")
    assert isinstance(threshold, float)
    assert gr > 0
    assert 1 < threshold < 5
