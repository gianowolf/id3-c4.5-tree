import pandas as pd
from tree_algorithms.utils.split_dataset import split_dataset

def test_split_dataset_two_values():
    df = pd.DataFrame({
        "Hero": ["Invoker", "PA", "Invoker", "PA"],
        "Result": ["Win", "Lose", "Win", "Lose"]
    })
    splits = split_dataset(df, "Hero")
    assert set(splits.keys()) == {"Invoker", "PA"}

def test_split_dataset_three_values():
    df = pd.DataFrame({
        "Hero": ["Invoker", "PA", "CM"],
        "Result": ["Win", "Lose", "Lose"]
    })
    splits = split_dataset(df, "Hero")
    assert set(splits.keys()) == {"Invoker", "PA", "CM"}

def test_split_dataset_single_value():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "Invoker"],
        "Result": ["Win", "Lose", "Win"]
    })
    splits = split_dataset(df, "Hero")
    assert set(splits.keys()) == {"Invoker"}

def test_split_dataset_numeric():
    df = pd.DataFrame({
        "GPM": [500, 300, 600, 400],
        "Result": ["Win", "Lose", "Win", "Lose"]
    })
    threshold = 450
    splits = split_dataset(df, "GPM", threshold)
    assert "<= 450" in splits
    assert "> 450" in splits
    assert all(splits["<= 450"]["GPM"] <= threshold)
    assert all(splits["> 450"]["GPM"] > threshold)
