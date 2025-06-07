from tree_algorithms.utils.majority_class import majority_class
from tree_algorithms.utils.best_split import best_split_attribute
from tree_algorithms.utils.split_dataset import split_dataset

def build_tree(dataset, attributes, target_column, min_gain=0.0, method="gain_ratio", max_depth=None, depth=0):
    """
    Construye recursivamente un árbol de decisión usando C4.5 (gain_ratio).

    Parámetros:
    - dataset (pd.DataFrame): Conjunto de datos actual.
    - attributes (List[str]): Atributos restantes por evaluar.
    - target_column (str): Nombre de la columna objetivo.
    - min_gain (float): Umbral mínimo de ganancia para expandir el árbol.
    - method (str): Método de evaluación ('gain_ratio' o 'gain').
    - max_depth (int or None): Profundidad máxima del árbol. None = sin límite.
    - depth (int): Profundidad actual del nodo.

    Retorna:
    - dict o str: Árbol de decisión o clase hoja.
    """
    if len(dataset[target_column].unique()) == 1:
        return dataset[target_column].iloc[0]
    if not attributes:
        return majority_class(dataset, target_column)
    if max_depth is not None and depth >= max_depth:
        return majority_class(dataset, target_column)

    result = best_split_attribute(dataset, attributes, target_column, method=method)
    if result is None:
        return majority_class(dataset, target_column)

    best_attr, threshold, gain_value = result
    if gain_value < min_gain:
        return majority_class(dataset, target_column)

    tree = {best_attr: {}}
    attr_key = (best_attr, threshold) if threshold is not None else best_attr
    subsets = split_dataset(dataset, attr_key)

    for attr_value, subset in subsets.items():
        if subset.empty:
            tree[best_attr][attr_value] = majority_class(dataset, target_column)
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attr]
            tree[best_attr][attr_value] = build_tree(
                subset,
                remaining_attributes,
                target_column,
                min_gain=min_gain,
                method=method,
                max_depth=max_depth,
                depth=depth + 1
            )

    return tree


def predict(tree, sample):
    """
    Realiza una predicción sobre una muestra dada un árbol entrenado.

    Parámetros:
    - tree (dict o str): Árbol de decisión entrenado (nodo o clase).
    - sample (dict o pd.Series): Muestra a clasificar.

    Retorna:
    - Clase predicha (str).
    """
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))

    if isinstance(attribute, tuple):
        attr_name, threshold = attribute
        value = sample[attr_name]
        branch = f"<= {threshold}" if value <= threshold else f"> {threshold}"
    else:
        value = sample[attribute]
        branch = value

    subtree = tree[attribute].get(branch)
    if subtree is None:
        return None  # Política ante valores no vistos
    return predict(subtree, sample)
