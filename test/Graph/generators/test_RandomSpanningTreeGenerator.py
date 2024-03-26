from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from Graph.utils.ConstraintChecks import get_degree_constraint_check
import random


def test_spanning_tree_generator_in_unconstrained_setting():
    n = 30
    for _ in range(25):
        for method in ['broder', 'kruskal']:
            graph = GraphGenerator.add_edges_erdosrenyi(n, p = 0.6)
            assert graph.is_connected()
            tree = RandomSpanningTreeGenerator.sample(graph, method = method)
            assert tree.is_spanning_tree()
            assert tree.get_m() == (graph.get_n() - 1)


def test_pruefer_spanning_tree_generator():
    """Only for complete graphs."""
    n = 30
    for _ in range(50):
        graph = GraphGenerator.add_edges_complete(n)
        tree = RandomSpanningTreeGenerator.sample(graph, 'pruefer')
        assert tree.is_spanning_tree()
        assert tree.get_m() == (graph.get_n() - 1)


def test_degree_constraint_tree_generator_on_complete_graph():
    n = 30
    graph = GraphGenerator.add_edges_complete(n)
    for _ in range(25):
        for method in ['broder', 'kruskal']:
            max_degree = random.randint(3, 10)  # sample degree-constraint
            tree = RandomSpanningTreeGenerator.sample(
                graph, would_violate_constraint = get_degree_constraint_check(max_degree))
            assert tree.is_spanning_tree()
            assert tree.get_max_degree() <= max_degree
