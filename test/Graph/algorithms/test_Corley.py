from Graph.generators.GraphGenerator import *
from Graph.algorithms.Corley import MCMSTPrim


def test_corley_ehrgott():
    """
    tests a small example provided by Ehrgott (with exact solutions given)
    """

    n = 4
    graph = GraphGenerator.add_edges_empty(n)
    graph.q = 2

    graph.add_edge(Edge(1, 2, [1, 4]))
    graph.add_edge(Edge(1, 3, [1, 3]))
    graph.add_edge(Edge(2, 3, [2, 0]))
    graph.add_edge(Edge(2, 4, [0, 2]))
    graph.add_edge(Edge(3, 4, [3, 1]))

    c = MCMSTPrim(graph)
    c.solve()

    approx_set = c.get_approximation_set()
    approx_front = c.get_approximation_front()

    assert len(approx_set) == 3
    assert [3, 5] in approx_front
    assert [6, 4] in approx_front
    assert [2, 9] in approx_front


def test_corley_bossek():
    """
    tests a small example provided by Jakob (with exact solutions given)
    """

    n = 4
    graph = GraphGenerator.add_edges_empty(n)
    graph.q = 2

    graph.add_edge(Edge(1, 2, [28, 27]))
    graph.add_edge(Edge(1, 3, [59, 39]))
    graph.add_edge(Edge(2, 3, [31, 30]))
    graph.add_edge(Edge(2, 4, [61, 23]))
    graph.add_edge(Edge(3, 4, [57, 51]))
    graph.add_edge(Edge(1, 4, [15, 48]))

    c = MCMSTPrim(graph)
    c.solve()

    approx_set = c.get_approximation_set()
    approx_front = c.get_approximation_front()

    assert len(approx_set) == 3
    assert [74, 105] in approx_front
    assert [120, 80] in approx_front
    assert [107, 101] in approx_front
