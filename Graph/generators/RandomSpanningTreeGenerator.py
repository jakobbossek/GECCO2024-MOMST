import random
from Graph.Graph import Graph
from Graph.algorithms.MST import KruskalRelink
from EA.PrueferSequence import PrueferSequence
from Graph.utils.UnionFind import UnionFind

from typing import Callable


class RandomSpanningTreeGenerator:
    """Static methods for the generation of random spanning trees."""

    @staticmethod
    def sample(graph: Graph,
               method: str = 'broder',
               would_violate_constraint: Callable[[Graph], bool] = lambda tree, e: False) -> Graph:
        """
        Sample a spanning tree of a given connected graph.

        Args:
            graph (Graph): connected graph.
            method (str): Either 'broder' for the algorithm by Broder-Aldous,
            kruskal for a Kruskal-based approach, or
            'pruefer' for a Prüfer-sequence to graph conversion method.
            The default is 'broder' which is indeed the most efficient method.
            would_violate_constraint (function(Graph, Edge)): function that expects a
            Graph (tree) and an Edge object e and returns a Boolean. A return value
            True means that adding e to tree would violate a constraint on the tree.
            Default is no constraint, i.e., a function that always returns False.
            This parameter is ignored if method is 'pruefer'.
        Returns:
            Random spanning tree of the input graph.
        """
        if method not in ['broder', 'kruskal', 'pruefer']:
            raise ValueError(f"Method '{method}' for random spanning tree generation is unknown!")

        if method == 'pruefer':
            pcode = PrueferSequence.random_from_graph(graph)
            return PrueferSequence.sequence_to_tree(pcode, graph)

        if method == 'broder':
            return RandomSpanningTreeGenerator.sample_broder(graph, would_violate_constraint)

        if method == 'kruskal':
            return RandomSpanningTreeGenerator.sample_kruskal(graph, would_violate_constraint)

    @staticmethod
    def sample_kruskal(
        graph: Graph,
        would_violate_constraint: Callable[[Graph], bool] = lambda tree, e: False
    ) -> Graph:
        """
        Sample a random spanning tree.

        The algorithm [1] works by means of Kruskal's algorithm: an empty graph T
        is initialised. Next, the edges of the input graph are travered in random
        order. Let e = {v, w} be the current edge. If deg(v, T) < max_degree and
        deg(v, T) < max_degree and T remains acyclic by adding e, e is added to
        T. The procedure ends once a spanning tree is generated or all edges are
        processed and no feasible d-ST could be produced.

        [1] G. R. Raidl, An efficient evolutionary algorithm for the degree-constrained
        minimum spanning tree problem, In: Proceedings of the 2000 Congress on Evolutionary
        Computation. CEC, La Jolla, CA, USA, 2000, pp. 104-111 vol.1,
        doi: 10.1109/CEC.2000.870282.

        Args:
            graph (Graph): connected  graph.
            would_violate_constraint (function(Graph, Edge)): function that expects a
            Graph (tree) and an Edge object e and returns a Boolean. A return value
            True means that adding e to tree would violate a constraint on the tree.
            Default is no constraint, i.e., a function that always returns False.
        Returns:
            Random spanning tree of the input graph which satisfies the constraint(s).
        """
        # randomly shuffle the edges
        edges = graph.get_edges()
        random.shuffle(edges)
        return KruskalRelink(graph,
                             edges = edges,
                             would_violate_constraint = would_violate_constraint,
                             ).solve().get_solution()

    @staticmethod
    def sample_broder(graph: Graph,
                      would_violate_constraint = lambda tree, e: False,
                      max_iter: int | None = None) -> Graph | None:
        """
        Sample a spanning tree of a given connected graph using Broder's algorithm.

        Broder's algorithm from 1989 (independently introduced by Aldous in 1990)) samples a random start node v
        uniformly at random. It then performs a random walk starting at v. Edges {v, w} which correspond
        to the first visit of w are added to the growing tree.
        The method has a worst case time complexity of O(n^3) but on average runs in time O(n log n).
        Information: https://math.dartmouth.edu/~pw/math100w13/kothari.pdf

        Args:
            graph (Graph): connected graph.
            would_violate_constraint (function(Graph, Edge)): function that expects a
            Graph (tree) and an Edge object e and returns a Boolean. A return value
            True means that adding e to tree would violate a constraint on the tree.
            Default is no constraint, i.e., a function that always returns False.
            max_iter (int): Maximum number of iterations for the random walk. Defaults
            to n*n where n is the number of nodes in the graph. If the random walk
            does not succeed in finding a ST within this limit of steps, None is
            returned
        Returns:
            Random spanning tree of the input graph which satisfies the constraint(s).
        """
        assert graph.is_connected()

        n = graph.get_n()

        if max_iter is None:
            max_iter = n * n * n

        # initialise empty tree
        tree = graph.copy(copy_edges = False)

        # sample random starting node
        curr_node = random.randint(1, n)

        # now perform random walk until spanning tree is build
        uf = UnionFind(n)

        i = 0
        while not uf.is_one_set():  # and i < max_iter:
            i = i + 1

            # sample a random neighbor of the current node
            e = graph.get_random_adjacent_edge(curr_node)

            # unpack
            v, w = e.v, e.w

            # do random walk step
            curr_node = e.other(curr_node)

            if would_violate_constraint(tree, e):
                continue

            # skip edge if its addition would close a cycle
            if not uf.union(v, w):
                continue

            # add edge to (growing) MST
            tree.add_edge(e.copy(), check_exists = False)

        if tree.get_m() != (n - 1):
            return None

        return tree
