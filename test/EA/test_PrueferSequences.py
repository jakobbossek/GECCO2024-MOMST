from Graph.generators.GraphGenerator import GraphGenerator
from EA.PrueferSequence import PrueferSequence


def test_random_pruefer_code_generation():
    n = 20
    for _ in range(10):
        graph = GraphGenerator.add_edges_complete(n)

        # generate random Prüfer code
        pcode = PrueferSequence.random_from_graph(graph)
        assert len(pcode) == (graph.get_n() - 2)
        pcode_in_range = list(map(lambda e: e >= 1 and e <= n, pcode))
        assert all(pcode_in_range)

        # convert to spanning tree
        tree = PrueferSequence.sequence_to_tree(pcode, graph)
        assert tree.is_spanning_tree()

        # convert spanning tree back to Prüfer sequence and compare
        pcode2 = PrueferSequence.tree_to_sequence(tree)

        # check if codes are identical
        assert all(x == y for x, y in zip(pcode, pcode2))
