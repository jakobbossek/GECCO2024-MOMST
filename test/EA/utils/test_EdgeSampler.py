from EA.utils.EdgeSampler import EdgeSamplerDominationCount, do_fast_nondominated_sorting
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.Edge import Edge


def test_get_nondomination_ranks_on_positively_correlated_weights():
    n = 10
    points = [[i, i] for i in range(n)]

    ranks, dom_counter = do_fast_nondominated_sorting(points)

    # ranks are [1, 2, ..., n]
    assert ranks == list(range(1, n + 1))
    # domination counts are [0, 2, ..., n - 1]
    assert dom_counter == list(range(0, n))


def test_do_fast_nondominated_sorting_on_negatively_correlated_weights():
    n = 10
    points = [[i, n - i] for i in range(n)]

    ranks, dom_counter = do_fast_nondominated_sorting(points)

    # ranks are [1, 1, ..., 1]
    assert ranks == ([1] * n)
    # domination counts are [0, 0, ..., 0]
    assert dom_counter == ([0] * n)


def test_edge_sampling():
    n = 50
    r = 10
    graph = GraphGenerator.add_edges_erdosrenyi(n, p = 0.6)
    edge_list = graph.get_edges()

    es = EdgeSamplerDominationCount(edge_list)
    sampled_edges = es.sample(r)
    assert len(sampled_edges) == r
    assert all([isinstance(e, Edge) for e in sampled_edges])
