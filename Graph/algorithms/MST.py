"""Method to calculate the Minimum-Spanning-Tree (MST) of an undirected graph."""
# used to avoid problem with circular dependency (CC import Graph, Graph imports CC)
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from Graph.Graph import Graph
    from Graph.Edge import Edge

from Graph.algorithms.CC import CC
from Graph.utils.UnionFind import UnionFind
from Graph.algorithms.SingleObjectiveSolverInterface import SingleObjectiveSolverInterface
import random
import heapq


def assert_lambdas(lambdas: list[float] | None, graph: Graph) -> None:
    """Sanity check weight vector."""
    if lambdas is None and graph.is_multi_weighted():
        print("[Prim] Weight vector 'lambdas' required for multi-weighted graphs.")
        exit()
    if lambdas is not None:
        assert (abs(sum(lambdas) - 1) < 0.0000001)
        assert (len(lambdas) == graph.get_q())


class Prim(SingleObjectiveSolverInterface):
    """Implements Prim's Minimum Spanning Tree (MST) algorithm."""

    def __init__(self,
                 graph: Graph,
                 order: str = 'sorted',
                 would_violate_constraint: Callable[[Graph, Edge], bool] = (lambda tree, e: False)
                 ) -> None:
        """
        Initialise a Prim method object.

        Args:
            graph (Graph): source graph.
            order (str): either 'sorted' (classical Prim) or 'random'. The later
                consideres edges in random order in the tree growing process. This
                makes sense for random generation of spanning trees.
            would_violate_constraint (function(Graph, Edge)): function that expects a
                Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
        """
        assert graph.is_connected()
        assert order in ['sorted', 'random']

        self.graph = graph
        self.order = order
        self.would_violate_constraint = would_violate_constraint

    def solve(self, lambdas: list[float] | None = None) -> SingleObjectiveSolverInterface:
        """
        Calculate a Minimum Spanning Tree.

        Args:
            lambdas (list[float]): optional vector of weights for scalarisation.
                Defaults to None.
        """
        n = self.graph.get_n()

        assert_lambdas(lambdas, self.graph)

        # sample random start node
        start = random.randint(1, n)

        # easily identify visited nodes in constant time
        visited = [False] * (n + 1)
        visited[start] = True

        # initial empty tree
        self.tree = self.graph.copy(copy_edges = False)

        # edges of the form (cost of e, e) -> heapq.heapify sort by the first element
        edges = [(e.get_cost(lambdas), e) for e in self.graph.adjacency[start].values()]

        # build heap in time O(deg(start))
        heapq.heapify(edges)

        while edges and self.tree.get_m() < (n - 1):
            cost, e = heapq.heappop(edges)

            # TODO: this is ugly as hell and error-prone!
            # We need to fix this edge-order problem!
            # Acutally we should just identify w as the 'new' nodes
            # that is added to the growing ST and not need to
            # make a case distinction
            v, w = e.v, e.w
            # if both end points are already visited there is nothing to do
            if visited[v] and visited[w]:
                continue

            # check if the insertion of e = {v, w} would violate a constraint
            if self.would_violate_constraint(self.tree, e):
                continue

            # which node was added?
            # TODO: here we still need the case distinction
            to = v
            if visited[v]:
                to = w

            visited[v] = True
            visited[w] = True

            # add the edge to the growing tree
            self.tree.add_edge(e.copy(), check_exists = False)

            # iterate over the neighbors of the newly added node
            # and add non-visited nodes to the heap
            for e in self.graph.adjacency[to].values():
                cost, w = e.get_cost(lambdas), e.other(to)
                if not visited[w]:
                    heapq.heappush(edges, (cost, e))

        # Note: this can happen only in presence of degree constraints
        if self.tree.get_m() < (n - 1):
            self.tree = None

        assert self.tree.is_spanning_tree()

        return self

    def get_solution(self) -> Graph:
        """Return the spanning tree."""
        return self.tree


class KruskalRelink:
    """Kruskal-Relink method."""

    def __init__(self,
                 graph: Graph,
                 edges: list[Edge],
                 initial_tree: Graph | None = None,
                 would_violate_constraint: Callable[[Graph, Edge], bool] = (lambda tree, e: False)):
        """
        Initialise a KruskalRelink object.

        Args:
            graph (Graph): source graph.
            edges (list[Edge]): list of edges that shall be used as candidates for
                relinking.
            initial_tree (Graph | None): optionally pass an initial forest. If
                this is not None (the default value), the intial_tree is used
                to build a pre-initialised union-find data-structure. In other
                words: the edges in initial_tree are fixed.
            would_violate_constraint (function(Graph, Edge)): function that expects a
                Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
        """
        assert graph.is_connected()
        # assert isinstance(initial_tree, Graph)

        self.graph = graph
        self.initial_tree = initial_tree
        self.edges = edges
        self.would_violate_constraint = would_violate_constraint

    def solve(self):
        """Relink connected components."""
        n = self.graph.get_n()

        # number of relevant edges, i.e., those edges that will be considered for relinking
        m_relevant = len(self.edges)

        # If an initial tree is available, use it and construct associated UnionFind.
        # Otherwise create tree and UF anew.
        if self.initial_tree is not None:
            # if an initial tree (forest of sub-trees) is available, we also need an
            # initial UnionFind to proceed
            self.tree = self.initial_tree  # JB: need a copy?
            # get connected components of initial_tree
            ccs = CC(self.initial_tree)
            # create a new UnionFind data structure to store forrest after deletion (partial solution)
            uf = UnionFind(n, ccs.get_components())
        else:
            # prepare empty tree
            self.tree = self.graph.copy(copy_edges = False)
            # prepare empty UnionFind
            uf = UnionFind(n)

        # loop until all sets are merged
        i = 0
        while not uf.is_one_set() and i < m_relevant:
            e = self.edges[i]

            # get edge; unpack (v, w)
            v, w = e.v, e.w
            i = i + 1

            # check if addition of the edge would close
            if self.would_violate_constraint(self.tree, e):
                continue

            # skip edge if its addition would close a cycle
            if not uf.union(v, w):
                continue

            # add edge to (growing) MST
            self.tree.add_edge(e.copy(), check_exists = False)
            # print("added edge: (", v, ",", w, ")")

        # Note: this can happen only in presence of degree constraints
        if self.tree.get_m() < (n - 1):
            self.tree = None

        assert self.tree.is_connected()
        # assert self.tree.get_m() == (n - 1)

        return self

    def get_solution(self) -> Graph:
        """Return the spanning tree."""
        return self.tree


class Kruskal(SingleObjectiveSolverInterface):
    """Implements Kruskal's Minumum Spanning Tree (MST) algorithm."""

    def __init__(self,
                 graph: Graph,
                 initial_tree: Graph | None = None,
                 filter_edges: bool = False,
                 order: str = 'sorted',
                 would_violate_constraint: Callable[[Graph, Edge], bool] = (lambda tree, e: False)
                 ) -> None:
        """
        Initialise a Kruskal method object.

        Args:
            graph (Graph): source graph.
            initial_tree (Graph | None): optionally pass an initial forest. If
                this is not None (the default value), the intial_tree is used
                to build a pre-initialised union-find data-structure. In other
                words: the edges in initial_tree are fixed.
            order (str): either 'sorted' (classical Prim) or 'random'. The later
                consideres edges in random order in the tree growing process. This
                makes sense for random generation of spanning trees.
            filter_edges (bool): should edges be preprocessed in the sense that
                edges that would certainly close a cycle are being ignored before
                sorting takes place? Defaults to False. Only relevant if initial_tree
                is not None.
            would_violate_constraint (function(Graph, Edge)): function that expects a
                Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
        """
        assert graph.is_connected()
        if initial_tree is not None:
            assert initial_tree.is_forest()
        assert order in ['sorted', 'random']

        self.graph = graph
        self.initial_tree = initial_tree
        self.order = order
        self.filter_edges = filter_edges
        self.would_violate_constraint = would_violate_constraint

    def solve(self, lambdas: list[float] | None = None) -> SingleObjectiveSolverInterface:
        """
        Calculate a Minimum Spanning Tree.

        Args:
            lambdas (list[float]): optional vector of weights for scalarisation.
                Defaults to None.
        """
        n = self.graph.get_n()

        assert_lambdas(lambdas, self.graph)

        # get edges
        edges = self.graph.get_edges()

        # classical approacj
        if self.order == 'sorted':
            # keep only relevant edges (i.e., those whose end-points are in differnet CCs)
            if self.initial_tree is not None and self.filter_edges:
                edges = self.graph.get_relinking_edges(self.initial_tree)

            # sort edges in-place
            edges.sort(key = lambda e: e.get_cost(lambdas))
        elif self.order == 'random':
            random.shuffle(edges)  # in place shuffling

        relinker = KruskalRelink(self.graph, edges, self.initial_tree, self.would_violate_constraint)
        self.tree = relinker.solve().get_solution()

        # sanity checks
        assert self.tree.is_connected()
        assert self.tree.get_m() == (n - 1)

        return self

    def get_solution(self) -> Graph:
        """Return the spanning tree."""
        return self.tree
