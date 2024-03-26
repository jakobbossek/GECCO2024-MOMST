from Graph.generators.GraphGenerator import GraphGenerator


def test_generate_kc_m_correlated():
    n = 100
    f = 3
    fld, fud = 3, 5
    alpha = 0.5
    for _ in range(10):
        graph = GraphGenerator.generate_kc_m_correlated(n, f, fld, fud, alpha)
        assert graph.is_complete()
        assert graph.is_connected()
