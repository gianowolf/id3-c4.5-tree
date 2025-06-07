# main.py

import argparse
import sys
from tree_algorithms import id3, c45
from tree_algorithms.utils import discretize_numeric_features
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="ID3 and C4.5 decision tree implementation for Dota 2 match dataset"
    )
    
    parser.add_argument(
        "algorithm",
        type=str,
        help="Algorithm to use: [id3 | 3 | id] or [c4.5 | c4 | c45 | 4 | 45]"
    )

    parser.add_argument(
        "--bin-numeric",
        action="store_true",
        help="Discretize numeric attributes into categories (required for C4.5)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    id3_aliases = {"id3", "3", "id"}
    c45_aliases = {"c4.5", "c4", "c45", "4", "45"}

    algo = args.algorithm.lower()

    if algo in id3_aliases:
        use_algorithm = "id3"
    elif algo in c45_aliases:
        use_algorithm = "c4.5"
    else:
        print("Invalid algorithm name. Use one of: id3, 3, id, c4.5, c4, c45, 4, 45.")
        sys.exit(1)

    # Load dataset
    df = pd.read_csv("data/dota_matches.csv")

    # Handle numeric features if requested
    if use_algorithm == "c4.5" and args.bin_numeric:
        df = discretize_numeric_features(df, bins=3)  # or bins=2

    # Drop numeric attributes if not handled
    if use_algorithm == "id3" or (use_algorithm == "c4.5" and not args.bin_numeric):
        df = df.select_dtypes(include=["object"])

    target_column = "Result"
    attributes = [col for col in df.columns if col != target_column]

    if use_algorithm == "id3":
        tree = id3.build_tree(df, attributes, target_column)
    else:
        tree = c45.build_tree(df, attributes, target_column)

    print("\nGenerated decision tree:")
    print(tree)

if __name__ == "__main__":
    main()
