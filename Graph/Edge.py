import random
from typing import Self


class Edge:
    """Representation of an undirected and weighted edge."""

    def __init__(self, v: int, w: int, cost: list[float] | None = None) -> None:
        """
        Initialise an edge object.

        Args:
            v (int): first node.
            w (int): second node.
            cost (list[float]): vector of edge weights/costs. A length-one vector
                even if there is just a single weight per edge.
        """
        # assert isinstance(cost, list), 'Edge class: cost parameter must be a list!'
        self.v = v
        self.w = w

        if cost is not None:
            if not isinstance(cost, list):
                raise ValueError('Edge: costs need to be a list.')
        self.cost = cost if cost is not None else []

    def get_v(self) -> int:
        """Return first node of the edge."""
        return self.v

    def get_w(self) -> int:
        """Return second node of the edge."""
        return self.w

    def either(self, v: int) -> int:
        """
        Return the second node (v, w).

        Args:
            v (int): node ID of the source node.
        Returns:
            Integer: from (u, w) return w if u == v else return v.
        """
        if v == self.v:
            return self.w
        return self.v

    def other(self, w: int) -> int:
        """
        Return the second node (v, w).

        Args:
            v (int): node ID of the source node.
        Returns:
            Integer: from (u, w) return w if u == v else return v.
        """
        return self.either(w)

    def is_same(self, e: Self) -> bool:
        """
        Check wheter two edges are equal.

        Note: equality is determined via the incident nodes! Edge weights are
        not considered. I.e., two parallel edges with different edge weights
        are classified equal.

        Args:
            e (Edge): another edge.
        Returns:
            Boolean value: True if the end nodes are equal and False otherwise.
        """
        return (self.v == e.v and self.w == e.w) or (self.v == e.w and self.w == e.v)

    def get_cost(self, lambdas: list[float] | None = None) -> list[float]:
        """
        Get the cost of the edge.

        Args:
            lambdas (list[float] | None): an optional weight vector. Default
                is None.
        Returns:
            The cost of the edge. If lambdas is not None, the scalarised edge
                costs are returned.
        """
        if lambdas is None:
            return self.cost

        # assert (len(lambdas) == self.n)
        assert (abs(sum(lambdas) - 1) < 0.0000001)

        # return [sum([self.cost[i] * lambdas[i] for i in range(len(lambdas))])]
        return [sum(co * la for co, la in zip(self.cost, lambdas))]

    def copy(self) -> 'Edge':
        """Generate a true copy of an edge."""
        return Edge(self.v, self.w, self.cost)

    def __str__(self) -> str:
        """Get a string representation of an edge."""
        return f'{self.v}, {self.w}, {self.cost}'

    def __lt__(self, other: Self) -> bool:
        """
        Check if edge is less or equal to another edge (lt = less than).

        Note: this is a dummy operation!

        We need this since, e.g., the heapq.heapify() method used in
        Graph.algorithms.MST.Prim works with (edge_costs, edge)-tuples.
        If for two edges the edge costs are equal the edge objects are compared,
        but the heapq documentation says: "Tuple comparison breaks for
        (priority, task) pairs if the priorities are equal and the tasks do not
        have a default comparison order."

        Args:
            other (Edge): Other edge.

        Returns:
            True and False with equal probability.
        """
        return True if random.random() < 0.5 else False
