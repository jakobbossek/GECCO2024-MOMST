"""Calculate connected components in an undirected graph."""
# used to avoid problem with circular dependency (CC import Graph, Graph imports CC)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph.Graph import Graph

from collections import deque


class CC:
    """
    Class for the calculation of the connected components (CCs) of a graph.

    Args:
        graph (Graph): Input graph.
    Returns:
        Object of class CC.
    """

    def __init__(self, graph: Graph) -> None:
        """Initialise CC object."""
        # save the number of components
        self.cc_counter = 0

        # local utility variables
        q = deque()
        n = graph.get_n()

        # array to maintain CC numbers for each node
        self.cc = [None] * (n + 1)

        # outer loop over all nodes
        for s in range(1, n + 1):

            # skip if a CC was already assigned
            if self.cc[s] is not None:
                continue

            # otherwise increase number of CCs and start BFS from the node s
            self.cc_counter += 1
            self.cc[s] = self.cc_counter
            q.append(s)

            while len(q) > 0:
                # get next node
                v = q.popleft()

                # traverse neighborhood
                for w in graph.get_adjacent_nodes(v):

                    if self.cc[w] is None:
                        self.cc[w] = self.cc[v]
                        q.append(w)

    def count(self) -> int:
        """
        Get the number of connected components.

        Returns:
          Integer indicating the number of CCs.
        """
        return self.cc_counter

    def is_connected(self) -> bool:
        """
        Check the graph for connectedness.

        Returns:
          Boolean indicating whether the graph is connected.
        """
        return self.count() == 1

    def is_unconnected(self) -> bool:
        """
        Check the graph for unconnectedness.

        Returns:
          Boolean indicating whether the graph is unconnected.
        """
        return self.count() > 1

    def get_node_assignment(self, v: int) -> int:
        """
        Receive the ID of the connected component a node is assigned to.

        Args:
            v (int): Node ID.
        Returns:
            Integer ID of the connected component node v was assigned to.
        """
        return self.cc[v]

    def in_same(self, v: int, w: int) -> bool:
        """
        Check if two nodes are in the same connected component.

        Args:
            v (int): First node ID.
            w (int): Second node ID.
        Returns:
            Boolean indicating if both nodes are in the same connected component.
        """
        return self.cc[v] == self.cc[w]

    def get_components(self) -> list[list[int]]:
        """
        Receive the connected components.

        Returns:
          A nested list of lists. For each CC there is a sub-list with
          the node numbers of the nodes forming the CC.
        """
        comps: list[list[int]] = [[] for _ in range(self.cc_counter)]
        for i, cc in enumerate(self.cc):
            if i == 0:
                continue
            comps[cc - 1].append(i)

        return comps

    def get_counts(self) -> list[int]:
        """
        Receive a list of sizes of connected components.

        Returns:
          A list of integer values.
        """

        return [len(cc) for cc in self.get_components()]

    def __iter__(self):
        """Initialise connected component iterator."""
        self.comps = self.get_components()
        self.comp_index = 0
        return self

    def __next__(self):
        """Get next connected component."""
        if self.comp_index >= self.count():
            return StopIteration

        c = self.comps[self.comp_index]
        self.comp_index += 1
        return c
