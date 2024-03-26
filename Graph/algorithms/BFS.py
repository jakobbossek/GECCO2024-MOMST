# used to avoid problem with circular dependency (BFS import Graph, Graph imports BFS)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph.Graph import Graph

"""Breadth-First-Search with optional limited search depth."""
from collections import deque


class BFS:
    """Breadth-First-Search with optional limited search depth."""

    def __init__(
        self,
        graph: Graph,
        start: int = 1,
        max_visited: int | None = None,
        build_tree: bool = False
    ) -> None:
        """
        Perform (limited) Breadth-First-Search (BFS) algorithm for connected graphs.

        Here, 'limited' refers to the possibility to set a maximal number of nodes as
        a parameter 'max_visited'. Once 'max_visited' are queued, the exploration terminates.

        This behaviour is required for out sub-graph based deletion-first strategy.

        Args:
            graph (Graph): (weighted) graph.
            start (int): start node. Defaults to the first node.
            max_visited (int): limit for the number of nodes visited. Once max_visited nodes have been selected
            the algorithm terminates. Defaults to the number of nodes of the graph.
            build_tree (bool): should the edges be constructed alongside.

        Returns:
            Object of type BFS.
        """
        assert graph.is_connected()

        n = graph.get_n()

        assert 1 <= start <= n

        q = deque()

        # distances to start node
        self.levels = [None] * n
        self.max_level = 0

        # list of visited nodes (for convenience)
        # A node is marked as 'visited' once it has been taken out of the queue
        self.visited_nodes = []

        # list of queued nodes
        self.queued_nodes = []

        # dummy for tree
        self.bfs_tree = None
        if build_tree:
            self.bfs_tree = graph.copy(copy_edges = False)

        # prepare
        q.append(start)
        self.queued_nodes.append(start)

        # book-keeping
        n_visited = 0  # number of already visited (i.e., processed) nodes
        n_queued = 1  # holds the number of nodes which already landed in the queue

        self.levels[start - 1] = 0
        curr_level = 1  # distance to start node

        while len(q) > 0:
            # terminate outer loop if the maximum number of nodes has already been hit
            if max_visited is not None:
                if n_queued == max_visited:
                    break

            v = q.popleft()

            n_visited += 1
            self.visited_nodes.append(v)

            for e in graph.adjacency[v].values():
                w = e.other(v)

                # skip if node was already put into the queue
                if self.levels[w - 1] is not None:
                    continue

                # update queue
                q.append(w)
                n_queued += 1
                self.queued_nodes.append(w)

                self.levels[w - 1] = curr_level

                # update BFS tree
                if build_tree:
                    edge = graph.get_edge(v, w).copy()
                    self.bfs_tree.add_edge(edge)

                # terminate inner loop if the maximum number of nodes has already been hit
                if max_visited is not None:
                    if n_queued == max_visited:
                        break

            # increase level counter
            curr_level += 1

            # maximum number of levels
            self.max_level = max([x for x in self.levels if x is not None])

    def get_queued_nodes(self) -> list[int]:
        """
        Get at list of node IDs of queued nodes.

        Note that this matches the output of get_visited_nodes() if the parameter
        'max_visited' was set to None.

        Returns:
            List of node IDs.
        """
        return self.queued_nodes

    def get_visited_nodes(self) -> list[int]:
        """
        Get at list of node IDs of visited nodes, i.e., nodes that were extracted from the queue already.

        Returns:
            List of node IDs.
        """
        return self.visited_nodes

    def get_last_visited_node(self) -> int:
        """
        Get ID of the last visited node.

        Returns:
            Single integer value.
        """
        return self.visited_nodes[-1]

    def get_maximum_level(self) -> int:
        """
        Return maximum level, i.e., distance to start node.

        This BFS algorithm keeps track of the distance, measured in terms of
        the number of edges between the start node and any other node.
        This method returns the maximum distance.

        Returns:
            Single integer value representing the largest distance from the start
            node to any other node.
        """
        return self.max_level

    def get_nodes_by_levels(self) -> list[list[int]]:
        """
        Get the nodes level-wise.

        I.e., for each distance d = 0, 1, 2, ... from the start
        a list of nodes with distance d to the start node.

        Returns:
            A list of integer lists. The d-th entry of the outer list contains
            a list of node IDs with distance d to the start node.
        """
        return list([[x for x in self.visited_nodes if self.levels[x - 1] == i] for i in range(self.max_level + 1)])

    def get_tree(self) -> Graph:
        """
        Return the BFS-tree.

        Returns:
            A Graph object.
        """
        return self.bfs_tree
