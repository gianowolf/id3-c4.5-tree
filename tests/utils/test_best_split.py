import pandas as pd
from tree_algorithms.utils.best_split import best_split_attribute

def test_best_split_id3_simple():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Role": ["Mid", "Mid", "Carry", "Carry"],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    best_attr, _ = best_split_attribute(df, ["Hero", "Role"], "Result", method="gain")
    assert best_attr in {"Hero", "Role"}

def test_best_split_id3_mixed_gain():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "PA"],
        "Item": ["BKB", "BKB", "Blink", "Blink", "Blink"],
        "Result": ["Win", "Win", "Lose", "Lose", "Win"]
    })
    best_attr, _ = best_split_attribute(df, ["Hero", "Item"], "Result", method="gain")
    assert best_attr == "Hero"

def test_best_split_c45_gain_ratio():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "PA"],
        "Item": ["BKB", "BKB", "Blink", "Blink", "Blink"],
        "Result": ["Win", "Win", "Lose", "Lose", "Win"]
    })
    best_attr, _ = best_split_attribute(df, ["Hero", "Item"], "Result", method="gain_ratio")
    assert best_attr in {"Hero", "Item"}

def test_best_split_empty_attribute_list():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Lose", "Lose", "Win"]
    })
    result = best_split_attribute(df, [], "Result", method="gain")
    assert result is None
