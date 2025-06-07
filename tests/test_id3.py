import pandas as pd
from tree_algorithms.id3 import build_tree

def test_id3_simple_tree():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    tree = build_tree(df, ["Hero"], "Result")
    assert isinstance(tree, dict)
    assert "Hero" in tree
    assert tree["Hero"]["Invoker"] == "Win"
    assert tree["Hero"]["PA"] == "Lose"

def test_id3_nested_split():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "CM", "CM"],
        "Item": ["BKB", "Blink", "BKB", "Blink", "Blink", "BKB"],
        "Result": ["Win", "Win", "Lose", "Lose", "Lose", "Win"]
    })
    tree = build_tree(df, ["Hero", "Item"], "Result")
    assert isinstance(tree, dict)
    assert "Hero" in tree
    assert "CM" in tree["Hero"]
    nested = tree["Hero"]["CM"]
    assert isinstance(nested, dict)
    assert "Item" in nested
    assert nested["Item"]["BKB"] == "Win"
    assert nested["Item"]["Blink"] == "Lose"
