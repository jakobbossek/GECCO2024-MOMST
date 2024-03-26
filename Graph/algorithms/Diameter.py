# used to avoid problem with circular dependency (Diameter import Graph, Graph imports Diameter)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph.Graph import Graph

"""Methods for the calculation of the diameter of undirected graphs."""
from collections import deque


class TreeDiameter:
    """
    Calculates the diameter of a (spanning) tree graph.

    The diameter is defined as the number of edges on the longest shortest path
    between pairs of nodes. By definition the diameter is between 0 (only on
    a graph with no edges at all) and (V-1) on a line graph.

    This method is based on two consecutive DFS calls and thus takes only time
    O(V) since a (spanning tree) has only O(V) edges.
    See https://iq.opengenus.org/diameter-of-tree-using-dfs/ for details.

    Args:
      graph (Graph): input tree.

    Returns:
      Object of type TreeDiameter.
    """

    def __init__(self, graph: Graph) -> None:
        """Initialise TreeDiameter object."""
        assert graph.is_spanning_tree()
        self._calc_tree_diameter(graph)

    def _calc_tree_diameter(self, graph: Graph) -> None:
        # maximum found pair distance
        self.max_dist = -1
        # node that is at maximal distance from starting node
        self.max_dist_node = -1

        # Start first DFS call from first node (it does not matter where we start)
        # This stores the node that is maximally distant from node 1 in self.max_dist_node
        # -> self.max_dist_node is one end point of the longest shortest path
        self.visited = [False] * (graph.get_n() + 1)
        self.dfs(graph, 1, 0)
        v = self.max_dist_node

        # second DFS call from self.max_dist_node
        # This updates self.max_dist_node
        self.visited = [False] * (graph.get_n() + 1)
        self.dfs(graph, self.max_dist_node, 0)

        # set diameter
        self.diam = self.max_dist
        self.endpoints = (v, self.max_dist_node)

    def dfs(self, graph: Graph, v: int, d: float) -> None:
        """
        Recursive version of Depth-First-Search (DFS).

        This DFS implementation logs the node w farthest from the starting node v and the
        distance d(v, w) in terms of the number of edges in object variables.

        Args:
            G (Graph): source graph.
            v (int): ID of the start node.
            d (int): Maximum distace seen so far.

        Returns:
            Nothing. The method changes the object variables though.
        """
        # mark node as visited
        self.visited[v] = True

        # update distance
        if d > self.max_dist:
            self.max_dist = d
            self.max_dist_node = v

        for w in graph.get_adjacent_nodes(v):
            if not self.visited[w]:
                self.dfs(graph, w, d + 1)

    def get_diameter(self) -> int:
        """
        Return the length of the longest shortest path.

        Return:
            The integer diamter of the input graph.
        """
        return self.diam

    def get_endpoints(self) -> tuple[int, int]:
        """
        Return the end-points (start and end node) of the path which defines the diameter.

        Return:
            A tuple {v,w} where v and w are the IDs of the end-nodes of the longest
            shortest path in the input graph.
        """
        return self.endpoints


class Diameter:
    """
    Calculates the diameter of an arbitrary graph.

    The diameter is defined as the number of edges on the longest shortest path
    between pairs of nodes. By definition the diameter is between 0 (only on
    a graph with no edges at all) and (|V|-1) on a line graph.
    This method runs the BFS algorithm |V| times starting once from every node.
    Since BFS runs in time O(|V| + |E|) the total runtime is O(|V|^2 + |V|*|E|)
    = O(|V|^3)

    This class requires the graph to be connected.

    Args:
      graph (Graph): input graph.

    Returns:
      Object of type Diameter.
    """

    def __init__(self, graph: Graph) -> None:
        """Initialise Diameter object."""
        assert graph.is_connected()

        # stores the maximum pairwise distance found so far
        max_dist = 0
        n = graph.get_n()
        q = deque()

        # for each node run BFS in O(V + E)
        # Total runtime is thus V * O(V + E) = O(V^3)
        for s in range(1, n + 1):

            # keep track of already marked nodes
            dists = [None] * n
            dists[s - 1] = 0
            q.append(s)

            while q:
                v = q.popleft()

                for w in graph.get_adjacent_nodes(v):
                    if dists[w - 1] is None:
                        dists[w - 1] = dists[v - 1] + 1
                        q.append(w)
                        if dists[w - 1] > max_dist:
                            max_dist = dists[w - 1]

        # finally set the diameter
        self.diam = max_dist

    def get_diameter(self) -> int:
        """
        Return the length of the longest shortest path.

        Return:
            The integer diamter of the input graph.
        """
        return self.diam
