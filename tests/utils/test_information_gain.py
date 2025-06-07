import pandas as pd
from pytest import approx
from tree_algorithms.utils.information_gain import information_gain

def test_information_gain_balanced():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Lose", "Win", "Lose"]
    })
    gain = information_gain(df, "Hero", "Result")
    # Entropía total: 1.0, entropía condicional: 1.0 → gain = 0.0
    assert gain == approx(0.0, abs=0.001)

def test_information_gain_pure_split():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    gain = information_gain(df, "Hero", "Result")
    # Entropía total: 1.0, condicional: 0.0 → gain = 1.0
    assert gain == approx(1.0, abs=0.001)

def test_information_gain_skewed_split():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose", "Win"]
    })
    gain = information_gain(df, "Hero", "Result")
    # Entropía total: ~0.971, condicional: ~0.551 → gain ~0.42
    assert gain == approx(0.42, abs=0.01)

def test_information_gain_single_value():
    df = pd.DataFrame({
        "Hero": ["Invoker"] * 4,
        "Result": ["Win", "Lose", "Lose", "Win"]
    })
    gain = information_gain(df, "Hero", "Result")
    # No split = gain = 0
    assert gain == approx(0.0, abs=0.001)
