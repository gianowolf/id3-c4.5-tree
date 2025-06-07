from tree_algorithms.utils.best_split import best_split_attribute
from tree_algorithms.utils.split_dataset import split_dataset
from tree_algorithms.utils.majority_class import majority_class

def build_tree(dataset, attributes, target_column):
    if len(dataset[target_column].unique()) == 1:
        return dataset[target_column].iloc[0]
    if not attributes:
        return majority_class(dataset, target_column)

    best_attr, threshold = best_split_attribute(dataset, attributes, target_column, method="gain")
    if best_attr is None:
        return majority_class(dataset, target_column)

    tree = {best_attr: {}}
    subsets = split_dataset(dataset, best_attr, threshold)

    for attr_val, subset in subsets.items():
        if subset.empty:
            tree[best_attr][attr_val] = majority_class(dataset, target_column)
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][attr_val] = build_tree(subset, remaining_attrs, target_column)

    return tree
