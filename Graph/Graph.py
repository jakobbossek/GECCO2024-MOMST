"""Basic class for (multi-)edge-weighted graphs."""
import random
import os
from Graph.Edge import Edge
from Graph.algorithms.CC import CC
from collections import deque
import matplotlib.pyplot as plt
from EA.utils.EdgeSampler import do_fast_nondominated_sorting


class Graph:
    """
    Class for undirected (multi-)weighted graphs.

    Args:
        n (int): number of nodes.
        q (int): number of costs per node. Default is 0.
    Returns:
        Object of class Graph.
    """

    def __init__(self, n: int, q: int = 0) -> None:
        """Init Graph object."""
        self.n: int = n
        self.m: int = 0
        self.q: int = q
        self.costs: list[float] = [0] * self.q  # sum of edge costs

        # Only for Euclidean graphs
        # TODO: make class EuclideanGraph which inherits from Graph?
        self.points: list = None

        # Adjacency list are realised as dictionaries for quick edge inserting,
        # deletion and lookup
        self.adjacency: list[dict] = [{} for _ in range(n + 1)]

        # list of node degrees
        self.deg: list[int] = [0] * (n + 1)

        # QUICK ACCESS
        # ===
        # The following data structures are initialised and memorised
        # once certain methods are called. There are invalidated once
        # (after memorization) other event take place, e.g., a new edge
        # is added

        # Another adjacancy list for quick sampling of neighbour nodes.
        self.neighbors: list[list[int]] | None = None

        # List of Edge objects
        self.edges: list[Edge] = None

    def _dememorize(self) -> None:
        """Forget utility data-structures for 'quick access'."""
        self.edges = None
        self.neighbors = None

    def has_edge(self, v: int, w: int) -> bool:
        """
        Check whether an edge {v,w} given by integers v and w exists.

        Args:
            v (int): Number of first node.
            w (int): Number of second node.
        Returns:
            Boolean (True if edge exists).
        """
        return (w in self.adjacency[v]) or (v in self.adjacency[w])

    def get_edge(self, v: int, w: int) -> Edge | None:
        """
        Get edge object given its to incident edges (if such an edge exists).

        Args:
            v (int): Number of first node.
            w (int): Number of second node.
        Returns:
            Edge object if edge exists or None otherwise.
        """
        assert 1 <= v <= self.n
        assert 1 <= w <= self.n

        e = self.adjacency[v].get(w)
        if e is None:
            e = self.adjacency[w].get(v)

        return e

    def copy(self, copy_edges: bool = True):
        """
        Create a copy of a graph object.

        Args:
            copy_edges (bool): Should the edges be copied? Default is True.
        Returns:
            Exact copy of the source graph.
        """
        graph = Graph(self.get_n(), self.get_q())

        if copy_edges:
            for e in self.get_edges():
                graph.add_edge(e.copy(), False)

        if self.points is not None:
            graph.points = self.points.copy()

        return graph

    def delete_edge(self, v: int, w: int) -> bool:
        """
        Delete an edge given its to incident nodes.

        Args:
            v (int): First node.
            w (int): Second node.
        Returns:
            Boolean: True if edge was found and deleted and False otherwise.

        Note: this is currently a quite expensive operation since deleting an
        element from the adjacency list costs O(n).
        """
        deleted = False

        if self.has_edge(v, w):
            del self.adjacency[v][w]
            del self.adjacency[w][v]
            deleted = True

        if deleted:
            self.deg[w] -= 1
            self.deg[v] -= 1
            self.m -= 1
            self._dememorize()

        return deleted

    def add_edge(self, e: Edge, check_exists: bool = True) -> bool:
        """
        Add edge given its to incident nodes and a vector of costs.

        Args:
            e (Edge): Edge to be added.
            check_exists (Boolean): Inject edge only if it does not yet exist.
        Returns:
            Boolean indicating whether the edge was added or not. This can be
            False only if the edge already exists and check_exists = True.
        """
        v, w = e.v, e.w

        if check_exists:
            if self.has_edge(v, w):
                return False

        self.adjacency[v][w] = e
        self.adjacency[w][v] = e
        self.deg[v] += 1
        self.deg[w] += 1
        self.m += 1
        self._dememorize()

        return True

    def get_adjacent_nodes(self, v: int) -> list[int]:
        """
        Get a list of nodes adjacent to node v.

        Args:
            v (int): Number of first node.
        Returns:
            List of adjacent node IDs.
        """
        # TODO: here we first extract the entire set even if we terminate earlier!
        # Use something like yield?
        assert 1 <= v <= self.n

        return [w for w in self.adjacency[v].keys()]

    def get_random_adjacent_edge(self, v: int) -> int | None:
        """
        Sample a random adjacent edge of a given node.

        Args:
            v (int): Node number.
        Returns:
            Edge object.
        """
        if self.is_isolated(v):
            return None

        rand_pos = random.randint(0, self.deg[v] - 1)

        # build 'quick access' data structure
        if self.neighbors is None:
            self.neighbors = [[e for e in self.adjacency[v].values()] for v in range(0, self.n + 1)]

        return self.neighbors[v][rand_pos]

    def get_edges(self, nodes: list[int] | None = None) -> list[Edge]:
        """
        Return a list of edges of the Graph.

        Args:
            nodes (list[int]): optional list of nodes that shall be
            considered when extracting the edges. Default is all nodes.
            If just a subset is given all edges in the respective adjacency
            lists are returned.

        Returns:
            List of Edge objects.
        """
        only_relevant = nodes is not None

        if not only_relevant and self.edges is not None:
            return self.edges

        # by default we consider all neighborhoods
        if not only_relevant:
            nodes = range(1, self.n + 1)

        edges = []
        for v in nodes:
            for e in self.adjacency[v].values():
                # avoid duplicates
                if ((e.v == v) and (e.w > v)) or ((e.w == v) and (e.v > v)):
                    edges.append(e)

        if only_relevant:
            return edges

        self.edges = edges

        return self.edges

    def get_active_nodes(self):
        """
        Return a list, which contains nodes that are incident to at least one edge. Thus, the list comprises all
        "active" nodes, which contribute to the graph structure.
        """
        nodes = set()
        edges = self.get_edges()
        for edge in edges:
            nodes.add(edge.get_v())
            nodes.add(edge.get_w())

        return list(nodes)

    def is_equal_to(self, other_graph):
        """
        Return, whether another graph is equivalent to the current graph. Equivalence is measured based on the
        number of nodes and edges as well as on the detailed level of concrete edges and their cost vectors.

        Args:
            other_graph (Graph): graph instance, for which equivalence to the current graph shall be tested.
        """

        # basic checks for number of vertices (n), number of edges (m), and cost dimension (q)
        if self.n == other_graph.n and self.m == other_graph.m and self.q == other_graph.q:
            own_edgelist = self.get_edges()

            # check edge similarity (similar edges has to be == m)
            similar_edges = 0

            for e in own_edgelist:
                if other_graph.has_edge(e.v, e.w):
                    if e.get_cost(None) == other_graph.get_edge(e.v, e.w).get_cost(None):
                        similar_edges = similar_edges + 1
                else:
                    # edge is missing --> return False
                    return False
            if similar_edges == self.m:
                # all similar --> return True
                return True
        else:
            # basic check failed --> return False
            return False

    def get_relinking_edges(self, subgraph: 'Graph', edge_list: list[Edge] | None = None) -> list[Edge]:
        """
        Return a list of all relinking edges given an sub-graph.

        I.e., if the subgraph is unconnected, a list will be returned with edges
        {u,v} such that u and v are in different connected components of the subgraph.

        Args:
            subgraph (Graph): Subgraph of the graph object.
            edge_list: Optional list of edges. Default is None. In this case all edges of the source graph are considered.
        Returns:
            List of Edge objects.
        """
        comps = CC(subgraph)
        edges = self.get_edges() if edge_list is None else edge_list
        return [e for e in edges if not comps.in_same(e.v, e.w)]

    def get_n(self) -> int:
        """
        Getter for the number of nodes.

        Returns:
            The number of nodes.
        """
        return self.n

    def get_m(self) -> int:
        """
        Getter for the number of edges.

        Returns:
            The number of edges.
        """
        return self.m

    def get_q(self) -> int:
        """
        Getter for the number of costs (per node).

        Returns:
            The number of costs (per node).
        """
        return self.q

    def is_multi_weighted(self) -> bool:
        """
        Check if the graph has multiple weights per edge.

        Returns:
            Boolean indicating whether the graph is multi-weighted.
        """
        return self.get_q() > 1

    def get_sum_of_edge_costs(self) -> list[float]:
        """
        Getter for the overall component-wise sum of costs.

        Returns:
            List of the sum of edge costs.
        """
        # TODO: ugly as sin!
        s = [0] * self.q
        for e in self:
            s = [s[i] + e.get_cost()[i] for i in range(self.q)]
        return s

    def get_deg(self, v: int) -> int:
        """
        Getter for the degree of a node v.

        Args:
            v (int): ID of node.
        Returns:
            The degree of the given node.
        """
        assert 1 <= v <= self.n

        return self.deg[v]

    def get_max_degree(self) -> int:
        """
        Getter for maximum node degree.

        Returns:
            Maximum node degree.
        """
        return max(self.deg)

    def get_number_of_nodes_with_max_degree(self) -> int:
        """
        Getter for the number of nodes with the maximum node degree.

        Returns:
            Number of nodes with the maximum node degree.
        """
        max_degree = self.get_max_degree()
        return sum([int(self.get_deg(v) == max_degree) for v in range(1, self.get_n() + 1)])

    def violates_degree_constraint(self, max_degree: int) -> bool:
        """
        Check if the graph violates a constraint on the maximum degree.

        Returns:
            Boolean: True if at least one node violates the degree constraint.
        """
        assert 1 <= max_degree <= self.n
        return self.get_max_degree() > max_degree

    def get_number_of_leafs(self) -> int:
        """
        Getter for the number of leaf nodes of a graph, i.e., those nodes with node degree 1.

        Returns:
            The number of leafs.
        """
        leafs = [int(self.is_leaf(v)) for v in range(1, self.get_n() + 1)]
        return sum(leafs)

    def is_leaf(self, v: int) -> bool:
        """
        Check whether a node is a leaf.

        Args:
            v (int): ID of node.

        Returns:
            Boolean: True if node is a leaf node (i.e., it has degree one).
        """
        assert 1 <= v <= self.n

        return self.get_deg(v) == 1

    def get_leafs(self) -> list[int]:
        """
        Get a list of the leafs / leaf nodes, i.e., those nodes with degree 1.

        Returns:
            List[int]: list of integers (the leaf IDs).
        """
        return [v for v in range(1, self.get_n() + 1) if self.is_leaf(v)]

    def get_isolated_nodes(self):
        """
        Return a list of isolated nodes.

        Returns:
            list(int): list of node ids.
        """
        return [v for v in range(1, self.get_n()) if self.is_isolated(v)]

    def is_isolated(self, v: int) -> bool:
        """
        Check whether a node is isolated.

        Args:
            v (int): ID of node.

        Returns:
            Boolean: True if node is isoloted (i.e., it has degree zero).
        """
        assert 1 <= v <= self.n

        return self.get_deg(v) == 0

    def is_complete(self) -> bool:
        """
        Check whether the graph is complete.

        Returns:
            Boolean: True if the graph is complete.
        """
        return self.m == ((self.n * (self.n - 1)) / 2)

    def is_connected(self) -> bool:
        """
        Check whether the graph is connected.

        Returns:
            Boolean: True if the graph is connected and False otherwise.
        """
        return CC(self).is_connected()

    def is_spanning_tree(self) -> bool:
        """
        Check whether the graph is a spanning tree.

        Returns:
            Boolean: True if the graph is a spanning tree and False otherwise.
        """
        return CC(self).is_connected() and (self.m == (self.n - 1))

    def is_empty(self) -> bool:
        """
        Check if the graph has no edges at all.

        Returns:
            Boolean: True if the graph has zero edges.
        """
        return self.get_m() == 0

    def is_acyclic(self, start: int = 1) -> bool:
        """
        Check if the graph is acyclic by a standard Depth-First-Search algorithm.

        Args:
            start (int): Start node (default is 1).

        Returns:
            Boolean: True if the graph is acyclic, and False otherwise.
        """
        n = self.get_n()

        visited = [False] * (n + 1)
        pi = [None] * (n + 1)

        # put start node on stack
        stack = deque()
        stack.append(start)
        pi[start] = start

        while len(stack) > 0:
            v = stack.pop()

            if visited[v]:
                continue

            visited[v] = True

            for e in self.adjacency[v].values():
                w = e.other(v)

                if not visited[w]:
                    pi[w] = v
                    stack.append(w)

                if visited[w] and not (w == pi[v]):
                    # cycle detected -> terminate
                    return False

        # no cycles detected
        return True

    def is_forest(self, exclude_special_cases: bool = False) -> bool:
        """
        Check if the graph is a forest.

        I.e., check if any two vertices are connected by
        at most one path. Put differently, a graph is a forest if all connected
        components are trees.

        Args:
            exclude_special_cases (bool): If True an edgeless graph and a
            spanning tree are not treated as a forest even though they fulfill
            the requirements. Default is False.
        """
        k = CC(self).count()
        ccs_are_trees = (self.get_n() - k) == self.get_m()
        if exclude_special_cases:
            # Special cases are:
            #   - a single (spanning) tree
            #   - an empty tree
            return ccs_are_trees and (k > 1) and (self.get_m() > 0)
        return ccs_are_trees

    def __iter__(self):
        """Initialise edge iterator."""
        self.edges = self.get_edges()
        self.edge_index = 0
        return self

    def __next__(self):
        """Get next edge."""
        if self.edge_index >= self.m:
            raise StopIteration

        e = self.edges[self.edge_index]
        self.edge_index += 1
        return e

    def plot(self) -> None:
        """Visualise Euclidean graph."""
        for e in self.get_edges():
            v, w, _ = e.v, e.w, e.get_cost()
            xs = [self.points[v - 1][0], self.points[w - 1][0]]
            ys = [self.points[v - 1][1], self.points[w - 1][1]]
            plt.plot(xs, ys, 'k-')

        # draw points
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        plt.plot(xs, ys, 'ro')

        plt.axis('equal')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.show()

    def plot_edges(self) -> None:
        """Plot scatter-plot of edge-weights of an bi-weighted graph."""
        if self.get_q() != 2:
            print('Scatterplot of edges supported only for bi-weighted graphs.')
            return

        # extract data
        edges = self.get_edges()
        xs = [e.get_cost()[0] for e in edges]
        ys = [e.get_cost()[1] for e in edges]

        # draw points
        plt.plot(xs, ys, 'o')
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

    def __str__(self) -> str:
        """Convert to string."""
        s = f'Graph object:\n{self.n} nodes\n{self.m} edges\n{self.q} weights per edge)'
        return s

    def is_dominated(self, other: 'Graph'):
        """ checks, whether a given graph dominates the self (Graph) w.r.t. edge costs. Returns True, if the
        self (Graph) is dominated."""
        graph_values = [self.get_sum_of_edge_costs(), other.get_sum_of_edge_costs()]
        ranks, _ = do_fast_nondominated_sorting(graph_values)
        return ranks[0] > 1

    @staticmethod
    def import_grapherator_format(path_to_file: str) -> 'Graph':
        """
        Import a multi-weighted graph from files generated with the multi-step graph generator grapherator in R.

        Args:
            path_to_file (str): Absolute path to the file.
        Returns:
            Graph object.
        """
        if not os.path.isfile(path_to_file):
            print(f'File path {path_to_file} does not exist!')

        graph = None
        with open(path_to_file, 'r') as fp:
            # first line contains the information on the number of nodes, edges, etc.
            n, m, c, q = list(map(int, fp.readline().strip().split(',')))
            graph = Graph(int(n), int(q))

            # Next three lines contain information the type of graph (skipped for now)
            fp.readline()
            fp.readline()
            fp.readline()

            # Next V lines contain the node coordinates
            graph.points = []
            for _ in range(n):
                x, y = list(map(float, fp.readline().split(',')))
                graph.points.append((x, y))

            # Remaining lines contain the edges: u, v, weight_1, ..., weight_W
            for _ in range(m):
                # read line and split
                line = fp.readline().split(',')
                # extract the node IDs
                v, w = int(line[0]), int(line[1])

                # avoid duplicates
                if w < v:
                    continue

                # generate weight vector
                cost = list(map(float, line[2:]))
                graph.add_edge(Edge(v, w, cost), check_exists = False)

        return graph

    def export_grapherator_format(self, graph: 'Graph', path_to_file: str) -> bool:
        """
        Export a multi-weighted graph to a ‘grapherator' file.

        Args:
            graph (Graph): source graph.
            path_to_file (str): Absolute path to the file.
        Returns:
            Graph object.
        """
        bn = os.path.dirname(path_to_file)
        if not os.path.isdir(bn):
            print(f'Path {bn} does not exist!')
            return False

        with open(path_to_file, 'w') as fp:
            # 0 = number of clusters
            fp.write(f'{graph.get_n()},{graph.get_m()},0,{graph.get_q()}\n')
            fp.write(', '.join(['RWG'] * graph.get_q()) + '\n')  # dummy
            fp.write('UNG\n')  # dummy
            fp.write('CEG\n')  # dummy

            # write points
            for x, y in graph.points:
                fp.write(f'{x},{y}\n')

            for e in graph:
                v, w, costs = e.v, e.w, e.get_cost()
                fp.write(','.join([str(v), str(w)] + [str(ec) for ec in costs]) + '\n')

        return True

    @staticmethod
    def get_subgraph_by_deleting_random_edges(graph: 'Graph', n_edges: int) -> 'Graph':
        """
        Obtain a subgraph of 'graph' by deleting 'n_edges' random edges.

        Note: this method runs in time O(n + m) where n and m is the number of
            nodes and edges of 'graph' respectively. If 'graph' is a (spanning)
            tree it runs in time O(n).
        Note: this method does not modify 'graph'. Instead, it returns a new
            graph.

        Args:
            graph (Graph): the input graph.
            n_edges (int): the number of edges to be deleted.

        Returns:
            Graph: a subgraph of the input graph.
        """
        # get list of all edges and permute randomly
        edges = graph.get_edges()
        random.shuffle(edges)

        # initialise new graph
        subgraph = graph.copy(copy_edges = False)

        # add the first (m - n_edges)
        for i in range(graph.get_m() - n_edges):
            subgraph.add_edge(edges[i].copy(), check_exists = False)

        return subgraph

    @staticmethod
    def get_graph_minus_graph(graph: 'Graph', subtract: 'Graph') -> 'Graph':
        """
        Obtain a subgraph of 'graph' by deleting all edges that are contained in
        graph and the subtract graph. This returns a subgraph of graph, which
        contains the edges of graph (set) minus the edges of subtract.

        Note: this method does not modify 'graph'. Instead, it returns a new
            graph.

        Args:
            graph (Graph): the input graph (from which the other graph is subtracted.
            subtract (Graph): the graph which is subtracted.

        Returns:
            Graph: a subgraph of the input graph (set) minus the subtract graph.
        """
        # get list of all edges
        edges_subtract = subtract.get_edges()

        # initialise new graph
        subgraph = graph.copy(copy_edges=False)

        # remove the edges from the subtracted graph
        for i in range(len(edges_subtract)):
            subgraph.delete_edge(edges_subtract[i].get_v(), edges_subtract[i].get_w())

        return subgraph

    @staticmethod
    def induced(graph: 'Graph', nodes: list[int]) -> 'Graph':
        """Get induced subgraph."""
        induced_graph = Graph(len(nodes), graph.get_q())

        # copy relevant node coordinates if available
        if graph.points is not None:
            induced_graph.points = [graph.points[v - 1] for v in nodes]

        # mapping from original node IDs to new node ids (1, ..., |nodes|)
        mapping = {x: i + 1 for i, x in enumerate(nodes)}

        # helper for quick lookup
        visited = [False] * (graph.get_n() + 1)
        for v in nodes:
            visited[v] = True

        # get all relevant edges
        # i.e., restricted to adjacency lists of selected nodes
        relevant_edges = graph.get_edges(nodes)
        for e in relevant_edges:
            v, w = e.v, e.w
            # add edge only if both end points are selected
            if visited[v] and visited[w]:
                # check_exists = False, since get_edges() makes sure that no back and forth edges are traversed
                induced_graph.add_edge(Edge(mapping[v], mapping[w], e.get_cost()), check_exists = False)

        return induced_graph

    @staticmethod
    def union(graph1: 'Graph', graph2: 'Graph', check_exists: bool = False) -> 'Graph':
        """
        Build union graph.

        Take two graphs and build the union-graph which contains all nodes
        and the union of the edge sets of both graphs.

        Args:
            graph1 (Graph): first graph.
            graph2 (Graph): second graph.
            check_exists (Boolean): If set to True duplicate edges are avoided in
            the union-graph.
        Returns:
            Graph object representing the union-graph.
        """
        union_graph = graph1.copy()
        for e in graph2:
            union_graph.add_edge(e, check_exists)

        return union_graph


