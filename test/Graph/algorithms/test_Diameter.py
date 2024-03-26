from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from Graph.algorithms.Diameter import Diameter, TreeDiameter


def test_on_line_graphs():
    n = 10
    for _ in range(10):
        graph = GraphGenerator.add_edges_line(n)
        diam = Diameter(graph).get_diameter()
        assert diam == (n - 1)


def test_on_complete_graphs():
    n = 10
    for _ in range(10):
        graph = GraphGenerator.add_edges_complete(n)
        diam = Diameter(graph).get_diameter()
        assert diam == 1


def test_on_star_graphs():
    n = 10
    for _ in range(10):
        graph = GraphGenerator.add_edges_star(n)
        diam = Diameter(graph).get_diameter()
        assert diam == 2


def test_tree_diameter_on_star_graphs():
    graph = GraphGenerator.add_edges_star(30)
    diam = TreeDiameter(graph).get_diameter()
    assert diam == 2


def test_tree_diameter_on_random_spanning_trees_of_a_complete_graph():
    graph = GraphGenerator.add_edges_complete(30)
    for _ in range(30):
        tree = RandomSpanningTreeGenerator.sample(graph)
        diam = TreeDiameter(tree).get_diameter()
        assert 2 <= diam <= tree.get_n()
