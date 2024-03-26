from Graph.Edge import Edge
from Graph.Graph import Graph
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from Graph.algorithms.MST import Kruskal


def test_general_graph_methods():
    n = 5
    graph = Graph(n, 1)
    assert not graph.has_edge(1, 2)
    assert not graph.has_edge(1, 3)
    graph.add_edge(Edge(1, 2, [1]))
    graph.add_edge(Edge(1, 3, [1]))
    graph.add_edge(Edge(1, 4, [1]))
    assert graph.get_deg(1) == 3
    assert graph.get_deg(2) == 1
    assert graph.get_deg(5) == 0
    assert graph.get_m() == 3
    graph.add_edge(Edge(1, 5, [1]))
    assert graph.has_edge(1, 2)
    assert not graph.is_leaf(1)
    assert graph.is_leaf(2)


def test_is_forest_on_empty_graph():
    graph = GraphGenerator.add_edges_empty(25)
    assert graph.is_forest()
    assert not graph.is_forest(True)


def test_is_forest_on_complete_graph():
    graph = GraphGenerator.add_edges_complete(25)
    assert not graph.is_forest()
    assert not graph.is_forest(True)


def test_is_forest_on_spanning_tree():
    graph = GraphGenerator.add_edges_complete(25)
    tree = Kruskal(graph).solve().get_solution()
    assert tree.is_forest()
    assert not tree.is_forest(True)


def test_is_empty():
    n = 10
    g = Graph(n, 1)
    assert g.is_empty()


def test_is_acyclic():
    graph = GraphGenerator.add_edges_star(10)
    assert graph.is_acyclic()

    graph = GraphGenerator.add_edges_complete(10)
    assert not graph.is_acyclic()
    tree = RandomSpanningTreeGenerator.sample(graph)
    assert tree.is_acyclic()

    graph = GraphGenerator.add_edges_line(10)
    assert graph.is_acyclic()


def test_grapherator_import():
    path_to_file = 'instances/CLASS1_100_4950_0_2_UNG_CEG_RWG-RWG_1.graph'
    graph = Graph.import_grapherator_format(path_to_file)
    assert graph.get_n() == 100
    assert graph.get_m() == 4950
    assert graph.get_q() == 2


def test_max_degree_on_star_graphs():
    n = 10
    graph = GraphGenerator.add_edges_star(n)
    max_degree = graph.get_max_degree()
    assert max_degree == (graph.get_n() - 1)
    # only the center node has maximum degree
    assert graph.get_number_of_nodes_with_max_degree() == 1


def test_max_degree_on_line_graph():
    n = 10
    graph = GraphGenerator.add_edges_line(n)
    max_degree = graph.get_max_degree()
    assert max_degree == 2
    # only the end nodes of the line have degree unequal to the maximum degree
    assert graph.get_number_of_nodes_with_max_degree() == (n - 2)


def test_get_number_of_leafs_line_graph():
    n = 10
    graph = GraphGenerator.add_edges_line(n)
    assert graph.get_number_of_leafs() == 2


def test_get_number_of_leafs_star_graphs():
    n = 10
    graph = GraphGenerator.add_edges_star(n)
    assert graph.get_number_of_leafs() == (n - 1)


def test_get_number_of_leafs_complete_graphs():
    n = 5
    graph = GraphGenerator.add_edges_complete(n)
    assert graph.get_number_of_leafs() == 0


def test_is_connected():
    n = 5
    graph = GraphGenerator.add_edges_line(n)
    graph.delete_edge(2, 3)
    assert not graph.is_connected()


def test_add_edge_exists():
    n = 5
    graph = Graph(n, 1)
    assert not graph.has_edge(1, 2)
    assert not graph.has_edge(1, 3)

    graph.add_edge(Edge(1, 2, [1]), check_exists=True)

    assert graph.has_edge(2, 1)
    assert len(graph.get_edges()) == 1

    graph.add_edge(Edge(1, 2, [1]), check_exists=True)

    assert len(graph.get_edges()) == 1


def test_delete_edge():
    n = 5
    graph = GraphGenerator.add_edges_line(n)
    graph.delete_edge(1, 2)
    graph.delete_edge(3, 2)

    assert not graph.has_edge(1, 2)
    assert not graph.has_edge(2, 3)


def test_is_complete():
    n = 4
    graph = Graph(n, 1)

    graph.add_edge(Edge(1, 2), True)
    graph.add_edge(Edge(1, 3), True)
    graph.add_edge(Edge(1, 4), True)
    graph.add_edge(Edge(1, 2), True)
    graph.add_edge(Edge(2, 3), True)
    graph.add_edge(Edge(2, 4), True)
    graph.add_edge(Edge(3, 4), True)

    assert graph.is_complete()


def test_is_isolated():
    n = 5
    graph = GraphGenerator.add_edges_star(n, center=1)
    for i in range(2, n + 1):
        graph.delete_edge(1, i)
        assert graph.is_isolated(i)


def test_union():
    n = 5
    # creates a complete graph out of the union of star graphs
    graph_init = GraphGenerator.add_edges_star(n)

    for i in range(2, n + 1):
        new_graph = GraphGenerator.add_edges_star(n, center=i)
        graph_init = Graph.union(graph_init, new_graph, check_exists=True)

    assert graph_init.is_complete()


def test_induced():
    n = 5

    cgraph = GraphGenerator.add_edges_complete(n)
    igraph = Graph.induced(cgraph, [1, 3, 5])
    assert igraph.is_complete()


def test_is_equal_to():
    n = 5

    # only graph structure
    cgraph1 = GraphGenerator.add_edges_complete(n)
    cgraph2 = GraphGenerator.add_edges_complete(n)

    assert cgraph1.is_equal_to(cgraph2)

    # test for different structure (star vs. complete)
    sgraph = GraphGenerator.add_edges_star(n)

    assert not cgraph1.is_equal_to(sgraph)

    # test for different weights
    cgraph1 = GraphGenerator.add_weights_random(cgraph1, q=2, min=0, max=10, asint=True)
    cgraph2 = GraphGenerator.add_weights_random(cgraph2, q=2, min=5, max=20, asint=True)

    assert not cgraph1.is_equal_to(cgraph2)
