"""Breadth-First-Search with optional limited search depth."""
# used to avoid problem with circular dependency (DFS import Graph, Graph imports CC)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph.Graph import Graph

from collections import deque


class DFS:
    """Depth-First-Search with optional limited search depth."""

    def __init__(
        self,
        graph: Graph,
        start: int = 1,
        max_visited: int | None = None,
        build_tree: bool = False
    ) -> None:
        """
        Perform (limited) Depth-First-Search (DFS) algorithm for connected graphs.

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

        # list of visited nodes (for convenience)
        # A node is marked as 'visited' once it has been taken out of the queue
        self.visited_nodes = []

        # list of queued nodes
        self.queued_nodes = []

        # dummy for tree
        self.dfs_tree = None
        if build_tree:
            self.dfs_tree = graph.copy(copy_edges = False)

        visited = [False] * (n + 1)
        pi = [None] * (n + 1)

        # put start node on stack
        stack = deque()
        stack.append(start)
        self.queued_nodes.append(start)
        pi[start] = start

        # book-keeping
        n_visited = 0  # number of already visited (i.e., processed) nodes
        n_queued = 1  # holds the number of nodes which already landed in the queue

        while len(stack) > 0:
            if max_visited is not None:
                if n_queued == max_visited:
                    break

            v = stack.pop()

            if visited[v]:
                continue

            n_visited += 1
            self.visited_nodes.append(v)
            visited[v] = True

            for e in graph.adjacency[v].values():
                w = e.other(v)

                # already visited -> skip
                if visited[w]:
                    continue

                # update data
                pi[w] = v

                # update BFS tree
                if build_tree:
                    self.dfs_tree.add_edge(e)

                stack.append(w)
                n_queued += 1
                self.queued_nodes.append(w)

                # terminate inner loop if the maximum number of nodes has already been hit
                if max_visited is not None:
                    if n_queued == max_visited:
                        break

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

    def get_tree(self) -> Graph:
        """
        Return the DFS-tree.

        Returns:
            A Graph object.
        """
        return self.dfs_tree
