import pandas as pd
from pytest import approx
from tree_algorithms.utils.attribute_entropy import attribute_entropy

def test_attribute_entropy_balanced():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Lose", "Win", "Lose"]
    })
    result = attribute_entropy(df, "Hero", "Result")
    # Cada grupo tiene entropía 1.0, ponderada por 0.5
    assert round(result, 3) == 1.000

def test_attribute_entropy_pure_groups():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose"]
    })
    result = attribute_entropy(df, "Hero", "Result")
    # Cada grupo tiene entropía 0.0
    assert result == 0.0

def test_attribute_entropy_skewed():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA", "PA"],
        "Result": ["Win", "Win", "Lose", "Lose", "Win"]
    })
    result = attribute_entropy(df, "Hero", "Result")
    # Entropía ponderada esperada ~0.550
    assert result == approx(0.551, abs=0.001)

def test_attribute_entropy_single_value():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "Invoker"],
        "Result": ["Win", "Lose", "Win"]
    })
    result = attribute_entropy(df, "Hero", "Result")
    # No hay división real; resultado igual a entropía general
    assert round(result, 3) == 0.918
