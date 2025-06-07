from graphviz import Digraph

def print_tree(tree, output_file="decision_tree", format="png", render=True):
    """
    Dibuja un árbol de decisión usando graphviz y lo exporta como imagen.

    Parámetros:
    - tree: dict
        Árbol de decisión en forma de diccionario anidado.
    - output_file: str
        Nombre base del archivo de salida.
    - format: str
        Formato de salida ('png', 'pdf', etc.).
    - render: bool
        Si True, renderiza y exporta la imagen. Si False, solo imprime en consola.
    """
    dot = Digraph(format=format)
    dot.attr(rankdir="TB", fontsize="12", fontname="Helvetica")
    dot.attr("node", shape="ellipse", style="filled", fillcolor="#E0F2F1", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    def add_nodes_edges(subtree, parent_id=None, edge_label=""):
        if not isinstance(subtree, dict):
            node_id = f"leaf_{id(subtree)}"
            dot.node(node_id, label=str(subtree), shape="box", style="filled", fillcolor="#B2DFDB", fontname="Helvetica-Bold")
            if parent_id:
                dot.edge(parent_id, node_id, label=edge_label)
            return

        attribute = next(iter(subtree))
        node_id = f"node_{id(subtree)}"
        dot.node(node_id, label=attribute)

        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)

        for value, child in subtree[attribute].items():
            add_nodes_edges(child, node_id, str(value))

    add_nodes_edges(tree)

    if render:
        dot.render(output_file, cleanup=True)
        print(f"Árbol exportado como {output_file}.{format}")
    else:
        print(dot.source)