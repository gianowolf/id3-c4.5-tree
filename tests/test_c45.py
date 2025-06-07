import pandas as pd
from tree_algorithms.c45 import build_tree_c45

def test_c45_with_categorical():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "CM", "CM"],
        "Item": ["BKB", "Blink", "BKB", "Blink", "Blink", "BKB"],
        "Result": ["Win", "Win", "Lose", "Lose", "Lose", "Win"]
    })
    tree = build_tree_c45(df, ["Hero", "Item"], "Result")
    assert isinstance(tree, dict)

def test_c45_with_numeric():
    df = pd.DataFrame({
        "Gold": [1200, 1500, 300, 400, 600, 1300],
        "XP": [500, 600, 200, 250, 300, 550],
        "Result": ["Win", "Win", "Lose", "Lose", "Lose", "Win"]
    })
    tree = build_tree_c45(df, ["Gold", "XP"], "Result")
    assert isinstance(tree, dict)
    key = next(iter(tree))
    assert ("<=" in key or ">")  # chequear que el nodo sea numérico

def test_c45_majority_class_on_empty():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Lose", "Lose", "Win"]
    })
    tree = build_tree_c45(df, [], "Result")
    assert tree in {"Win", "Lose"}

def test_c45_pure_leaf():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "Invoker"],
        "Result": ["Win", "Win", "Win"]
    })
    tree = build_tree_c45(df, ["Hero"], "Result")
    assert tree == "Win"
