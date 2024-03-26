from Graph.generators.GraphGenerator import GraphGenerator
from Graph.algorithms.MST import Kruskal, Prim
from Graph.utils.ConstraintChecks import get_degree_constraint_check


def test_on_complete_euclidean_graphs_with_one_weight_per_edge():
    n = 10
    for i in range(10):
        graph = GraphGenerator.generate_unit_weighted(n)
        mst1 = Kruskal(graph).solve().get_solution()
        mst2 = Prim(graph).solve().get_solution()

        assert mst1.is_spanning_tree()
        assert mst2.is_spanning_tree()

        # assure that both find the same tree
        # TODO: this works for single-weighted graphs only
        assert abs((mst1.get_sum_of_edge_costs()[0] - mst2.get_sum_of_edge_costs()[0]) < 0.000001)


def test_in_degree_constrained_setting():
    n = 30
    for _ in range(30):
        graph = GraphGenerator.generate_euclidean(n)
        # Planar MSTs have degree at most six. Thus we set 4 here
        max_degree = 4

        # test Prim's algorithm
        dmst = Prim(graph, would_violate_constraint = get_degree_constraint_check(max_degree)).solve().get_solution()
        assert dmst.is_spanning_tree()
        assert dmst.get_max_degree() <= max_degree

        # test Kruskal's algorithm
        dmst = Kruskal(graph, would_violate_constraint = get_degree_constraint_check(max_degree)).solve().get_solution()
        assert dmst.is_spanning_tree()
        assert dmst.get_max_degree() <= max_degree
