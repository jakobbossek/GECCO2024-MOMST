from Graph.generators.GraphGenerator import GraphGenerator
from Graph.algorithms.CC import CC


def test_on_star_graph():
    n = 10
    graph = GraphGenerator.add_edges_star(n)
    assert graph.get_n() == n
    assert graph.get_m() == n - 1
    ccs = CC(graph)
    assert ccs.count() == 1
    assert ccs.is_connected()
    assert not ccs.is_unconnected()
    assert len(ccs.get_components()) == 1


def test_on_empty_graph():
    n = 10
    graph = GraphGenerator.add_edges_empty(n)
    assert graph.get_n() == n
    assert graph.get_m() == 0
    ccs = CC(graph)
    assert ccs.count() == n
    assert not ccs.is_connected()
    assert ccs.is_unconnected()
    assert len(ccs.get_components()) == n
