"""Method for checking if adding and edge into a tree would violate constraints."""
from Graph.Edge import Edge
from Graph.Graph import Graph

from typing import Callable


def get_degree_constraint_check(max_degree: int) -> Callable[[Graph, Edge], bool]:
    """
    Get function which checks if adding an edge into a tree would violate a maximum degree constraint.

    Args:
        max_degree (int): Maximum node degree.
    Returns:
        Function(tree, e) that returns a Boolean.
    """
    def violates_constraint(tree: Graph, e: Edge) -> bool:
        v, w = e.v, e.w
        return tree.get_deg(v) >= max_degree or tree.get_deg(w) >= max_degree

    return violates_constraint
