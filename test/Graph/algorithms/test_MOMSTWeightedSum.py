from Graph.Graph import Graph
from Graph.algorithms.MST import Kruskal, Prim
from Graph.algorithms.Scalarisation import MOMSTWeightedSum
from EA.utils.EdgeSampler import do_fast_nondominated_sorting


def test_MOMSTWeightedSum_on_complete_bi_weighted_graph():
    path_to_file = 'instances/CLASS1_100_4950_0_2_UNG_CEG_RWG-RWG_1.graph'
    graph = Graph.import_grapherator_format(path_to_file)

    # equidistant weight vectors
    n_lambdas = 25
    lambdas = [(k / n_lambdas, 1 - k / n_lambdas) for k in range(n_lambdas + 1)]

    for method_class in [Prim, Kruskal]:
        solver = method_class(graph)
        momst_result = MOMSTWeightedSum(graph, lambdas = lambdas, solver = solver).solve()
        trees = momst_result.get_approximation_set()

        assert len(trees) == (n_lambdas + 1)
        assert all([tree.is_spanning_tree() for tree in trees])

        pf = momst_result.get_approximation_front()
        ranks, _ = do_fast_nondominated_sorting(pf)
        assert len(pf) == (n_lambdas + 1)

        # assert that all solutions are non-dominated
        assert all([r == 1 for r in ranks])
