import pandas as pd
from tree_algorithms.id3 import build_tree, predict
from tree_algorithms.utils.print_tree import print_tree

def main():
    df = pd.read_csv("data/meepo_dataset.csv")
    target = "saco_meepo_mid"
    features = [col for col in df.columns if col != target]

    # Usar gain_ratio (C4.5 style), detener si la ganancia es muy baja y limitar la profundidad del árbol
    tree = build_tree(
        dataset=df,
        attributes=features,
        target_column=target,
        min_gain=0.01,
        method="gain_ratio",
        allow_numeric=False,
        max_depth=3  # profundidad máxima permitida
    )

    if tree is None:
        print("No se pudo construir el árbol.")
        return

    print("\nÁrbol de decisión:")
    print_tree(tree, output_file="meepo_tree", format="png", render=True)

    # Mostrar ejemplo de predicción
    sample = df.iloc[0]
    print("\nEjemplo:")
    print("Predicción:", predict(tree, sample))
    print("Valor real:", sample[target])

if __name__ == "__main__":
    main()
