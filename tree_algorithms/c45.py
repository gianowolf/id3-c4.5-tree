from tree_algorithms.utils.majority_class import majority_class
from tree_algorithms.utils.best_split import best_split_attribute
from tree_algorithms.utils.split_dataset import split_dataset

def build_tree(dataset, attributes, target_column):
    """
    Recursively build a decision tree using the C4.5 algorithm.

    Parameters:
    - dataset (pd.DataFrame): The current dataset.
    - attributes (List[str]): Remaining attributes to consider.
    - target_column (str): The class label column.

    Returns:
    - dict or str: Nested dictionary representing the tree, or a leaf class.
    """
    # Caso base 1: todos los valores pertenecen a una sola clase
    if len(dataset[target_column].unique()) == 1:
        return dataset[target_column].iloc[0]

    # Caso base 2: no hay más atributos
    if not attributes:
        return majority_class(dataset, target_column)

    # Seleccionar el mejor atributo con Gain Ratio
    best_attr = best_split_attribute(dataset, attributes, target_column, method="gain_ratio")

    if best_attr is None:
        return majority_class(dataset, target_column)

    tree = {best_attr: {}}
    subsets = split_dataset(dataset, best_attr)

    for attr_value, subset in subsets.items():
        if subset.empty:
            tree[best_attr][attr_value] = majority_class(dataset, target_column)
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attr]
            subtree = build_tree(subset, remaining_attributes, target_column)
            tree[best_attr][attr_value] = subtree

    return tree
