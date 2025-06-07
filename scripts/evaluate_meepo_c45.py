import pandas as pd
from tree_algorithms.c45 import build_tree, predict
from tree_algorithms.utils.print_tree import print_tree
from tree_algorithms.utils.print_tree_cli import print_tree_cli

def main():
    df = pd.read_csv("data/meepo_dataset.csv")
    target = "saco_meepo_mid"
    features = [col for col in df.columns if col != target]

    # Construir árbol C4.5 con gain_ratio y atributos numéricos permitidos, y profundidad máxima
    tree = build_tree(
        dataset=df,
        attributes=features,
        target_column=target,
        min_gain=0.01,
        method="gain_ratio",
        max_depth=3  # profundidad máxima
    )


    if tree is None:
        print("No se pudo construir el árbol.")
        return

    print("\nÁrbol de decisión (C4.5):")
    print_tree(tree, output_file="meepo_tree_c45", format="png", render=True)

    print("\nÁrbol en consola:")
    print_tree_cli(tree)

    sample = df.iloc[0]
    print("\nEjemplo de predicción:")
    print("Predicción:", predict(tree, sample))
    print("Valor real:", sample[target])

if __name__ == "__main__":
    main()
