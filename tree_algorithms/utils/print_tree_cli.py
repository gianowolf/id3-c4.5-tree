import time
import sys

# ANSI escape codes
RESET = "\033[0m"
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
GRAY = "\033[90m"

# Determinar si es clase positiva o negativa (ajustable)
def is_positive_label(label):
    return str(label).lower() in {"yes", "sí", "true", "1"}

def is_negative_label(label):
    return str(label).lower() in {"no", "false", "0"}

def print_tree_cli(tree, indent="", is_last=True, delay=0.06):
    """
    Imprime un árbol de decisión con formato jerárquico, colores y efecto de animación.

    Parámetros:
    - tree (dict or str): Árbol de decisión o clase hoja.
    - indent (str): Indentación acumulada.
    - is_last (bool): Si el nodo actual es el último hermano.
    - delay (float): Tiempo entre líneas para efecto visual.
    """
    branch_prefix = "└── " if is_last else "├── "
    if not isinstance(tree, dict):
        # Colorear la clase según su valor
        if is_positive_label(tree):
            label = f"{GREEN}[Leaf] {tree}{RESET}"
        elif is_negative_label(tree):
            label = f"{RED}[Leaf] {tree}{RESET}"
        else:
            label = f"{GRAY}[Leaf] {tree}{RESET}"
        print(indent + branch_prefix + label)
        time.sleep(delay)
        return

    attribute = next(iter(tree))
    branches = tree[attribute]
    n = len(branches)

    print(indent + branch_prefix + f"{BLUE}[Attr] {attribute}{RESET}")
    time.sleep(delay)
    indent += "    " if is_last else "│   "

    for i, (branch_label, subtree) in enumerate(branches.items()):
        is_last_branch = (i == n - 1)
        print(indent + ("└── " if is_last_branch else "├── ") + f"{GRAY}[Val] {branch_label}{RESET}")
        time.sleep(delay)
        print_tree_cli(subtree, indent + ("    " if is_last_branch else "│   "), is_last_branch, delay)
