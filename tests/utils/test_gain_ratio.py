import pandas as pd
from pytest import approx
from tree_algorithms.utils.gain_ratio import gain_ratio

def test_gain_ratio_pure_split():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    ratio = gain_ratio(df, "Hero", "Result")
    # Ganancia = 1, IV = 1 → ratio = 1
    assert ratio == approx(1.0, abs=0.001)

def test_gain_ratio_no_information_gain():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Lose", "Win", "Lose"]
    })
    ratio = gain_ratio(df, "Hero", "Result")
    # Ganancia = 0, IV = 1 → ratio = 0
    assert ratio == approx(0.0, abs=0.001)

def test_gain_ratio_zero_iv():
    df = pd.DataFrame({
        "Hero": ["Invoker"] * 4,
        "Result": ["Win", "Lose", "Lose", "Win"]
    })
    ratio = gain_ratio(df, "Hero", "Result")
    # IV = 0 → gain_ratio debe ser 0
    assert ratio == 0.0

def test_gain_ratio_skewed():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose", "Win"]
    })
    ratio = gain_ratio(df, "Hero", "Result")
    # Se espera un ratio razonable entre 0 y 1
    assert 0.0 < ratio < 1.0
