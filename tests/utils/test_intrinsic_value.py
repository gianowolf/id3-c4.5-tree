import pandas as pd
from pytest import approx
from tree_algorithms.utils.intrinsic_value import intrinsic_value

def test_intrinsic_value_uniform_split():
    df = pd.DataFrame({
        "Hero": ["Invoker", "Invoker", "PA", "PA"]
    })
    result = intrinsic_value(df, "Hero")
    # Dos valores con 0.5 de probabilidad → IV = 1.0
    assert result == approx(1.0, abs=0.001)

def test_intrinsic_value_three_categories():
    df = pd.DataFrame({
        "Hero": ["Invoker", "PA", "CM", "Invoker", "PA", "PA"]
    })
    result = intrinsic_value(df, "Hero")
    # Tres valores, distribución no uniforme
    assert result == approx(1.459, abs=0.01)

def test_intrinsic_value_single_category():
    df = pd.DataFrame({
        "Hero": ["Invoker"] * 5
    })
    result = intrinsic_value(df, "Hero")
    # Sin división → IV = 0
    assert result == 0.0

def test_intrinsic_value_high_cardinality():
    df = pd.DataFrame({
        "Hero": [f"Hero_{i}" for i in range(10)]
    })
    result = intrinsic_value(df, "Hero")
    # Cada categoría es única → máxima IV
    assert result == approx(3.321, abs=0.01)  # log2(10)
