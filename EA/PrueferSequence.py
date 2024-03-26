"""Class with methods for the Pruefer-number encoding."""
import random
from Graph.Graph import Graph


class PrueferSequence:
    """
    Pruefer-sequence encoding.

    According to Cayley's formula any there is a bijection between the set of spanning trees
    of a complete graph with n nodes and the set of sequences of n-2 integers in the range {1, ..., n};
    these sequences are called Prüfer-codes or Prüfer-sequences.

    This class provides static methods to generate random Prüfer-sequences and transform
    Prüfer-codes to graphs and the vice versa.
    """

    @staticmethod
    def random_from_graph(graph: Graph) -> list[int]:
        """
        Create a random Prüfer-code from a complete graphs.

        Raises a ValueError for non-complete graphs.

        Args:
            graph (Graph): Complete source graph.
        Return:
            A list of integer values (the Prüfer sequence of T).
        """
        if not graph.is_complete():
            raise ValueError('G must be a complete graph!')
        return PrueferSequence.random(graph.get_n())

    @staticmethod
    def random(n: int) -> list[int]:
        """Create a random Pruefer sequence."""
        return [random.randint(1, n) for _ in range(n - 2)]

    @staticmethod
    def tree_to_sequence(tree: Graph) -> list[int]:
        """
        Generate the Prüfer sequence for a given labeled spanning tree T with vertices {1, ..., n}.

        Args:
            tree (Graph): A spanning tree T.
        Return:
            A list of integer values (the Prüfer sequence of T).

        Notes: See https://en.wikipedia.org/wiki/Prüfer_sequence for an explanation.
        """
        tree2 = tree.copy()
        n = tree.get_n()

        # initialise Prüfer sequence
        pcode = [0] * (n - 2)

        # TODO: this is inefficient due to deletion of edges and stuff
        # However, we do not need this operation at all at the moment (just
        # the inverse operation sequence_to_tree)
        for i in range(n - 2):

            # identify leaf node with lowest node ID among all leafs
            v = min(tree2.get_leafs())

            # now get the unique neighbor of the leaf with the lowest ID
            w = tree2.get_adjacent_nodes(v)[0]

            # TODO: inefficient
            tree2.delete_edge(v, w)

            # set ith pcode entry to v's unique neighbor
            pcode[i] = w

        return pcode

    @staticmethod
    def sequence_to_tree(pcode: list[int], graph: Graph) -> Graph:
        """
        Convert a Prüfer sequence of length n into a spanning tree with n+2 nodes.

        Args:
            pcode (list[int]): The Prüfer sequence, i.e., a list of n integers values all in the range {1, ..., n+2}.
            graph (Graph): the source graph.
        Return:
            A Graph object with n+2 nodes and n+1 edges.

        Notes: See https://en.wikipedia.org/wiki/Prüfer_sequence for an explanation.
        """
        n = len(pcode) + 2

        # initialise empty tree
        tree = Graph(n, graph.get_q())

        # initialise 'degrees' (with a dummy element at degrees[0])
        degrees = [1] * n

        # increase for each entry in pcode
        for v in pcode:
            degrees[v - 1] += 1

        # FIRST PHASE of tree construction: for each number in the sequence pcode[i],
        # find the first (lowest-numbered) node, j, with degree equal to 1, add the edge
        # (j, pcode[i]) to the tree, and decrement the degrees of j and pcode[i]
        for i in pcode:
            for j in range(1, n + 1):
                if degrees[j - 1] == 1:
                    # insert edge (i, j) into tree
                    tree.add_edge(graph.get_edge(i, j).copy(), check_exists = False)
                    degrees[i - 1] -= 1
                    degrees[j - 1] -= 1
                    break

        # SECOND PHASE of tree construction: two nodes with degree 1 are left between
        # two nodes u and v -> add {u, v}
        assert sum(degrees) == 2

        u, v = (i for i in range(1, n + 1) if degrees[i - 1] == 1)
        tree.add_edge(graph.get_edge(u, v).copy())

        assert tree.is_spanning_tree()

        return tree
