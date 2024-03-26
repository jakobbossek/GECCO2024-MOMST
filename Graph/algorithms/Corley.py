from EA.utils.EdgeOrderer import EdgeDominanceOrder
from EA.utils.EdgeSampler import do_fast_nondominated_sorting
from Graph.Graph import Graph
from Graph.algorithms.CC import CC
from Graph.algorithms.MultiObjectiveSolverInterface import MultiObjectiveSolverInterface


class MCMSTPrim(MultiObjectiveSolverInterface):

    """
    Implementation of Corley's algorithms for computing all efficient mcMST solutions. This
    implementation realises a corrected version of algorithm 9.4 from Ehrgott's book on
    Multicriteria Optimization [1].

    [1] Matthias Ehrgott, Multicriteria Optimization. Springer Berlin, Heidelberg, 2nd edition, 2005.
    https://doi.org/10.1007/3-540-27659-9


    Args:
        graph (Graph): the source graph on which the algorithm is applied.
    """
    def __init__(self, graph: Graph):
        assert graph.is_connected()

        self.graph = graph
        self.approxSet = []
        self.approxFront = []

    def solve(self) -> MultiObjectiveSolverInterface:
        """Solve the problem using Corley's algorithm. This method implements the main solver
        procedure. It implements a multi-objective version of Prim's algorithm.

        Main steps:
            (1) Find edges with non-dominated edge costs and use them to build a first set of partial solutions
            each including one of these non-dominated edges.
            (2) For each partial solution (tree) in the set of partial trees, continue to identify non-dominated
            edges, that extend the partial solutions towards more complete spanning trees
            (3) Finally, use non-dominated filtering to extract the efficient solutions and the Pareto front
        """

        n = self.graph.n
        partial_trees = []

        # Step (1):  identify non-dominated edges of the whole graph
        nondom_edges = EdgeDominanceOrder().order(self.graph.get_edges(), None)

        # Build first set of partial solutions (each partial tree contains only one non-dominated edge)
        for edge in nondom_edges:
            partial_tree = Graph(n, q=self.graph.get_q())
            partial_tree.add_edge(edge)
            partial_trees.append(partial_tree)

        # Step (2): successively generate more complete partial trees by extending each before created partial tree
        # this creates n-1 solution sets with partial trees, of which only the latest set is considered for the
        # next iteration.
        for _ in range(1, n - 1):
            new_partial_trees = []
            edges = self.graph.get_edges()

            for tree in partial_trees:
                # for each tree in the current partial tree set
                comps = CC(tree)
                considered_edges = []
                # determine considered edges for non-domination filtering
                for e in edges:
                    # only edges, which
                    #   - are not already included AND
                    #   - do not close a cycle in the tree AND
                    #   - are incident to an already connected node
                    # are considered.
                    if not tree.has_edge(e.v, e.w) and not comps.in_same(e.v, e.w) and not (tree.is_isolated(e.v) and tree.is_isolated(e.w)):
                        considered_edges.append(e)
                # the considered edges are filtered w.r.t. non-domination
                nondom_edges = EdgeDominanceOrder().order(considered_edges, None)
                # create new partial solutions for being considered during the next iteration
                for edge in nondom_edges:
                    n_tree = tree.copy()
                    n_tree.add_edge(edge)
                    new_partial_trees.append(n_tree)

                partial_trees = new_partial_trees

        # after computing all candidate solution of set n-1, compute their cost vector
        tree_values = [t.get_sum_of_edge_costs() for t in partial_trees]

        # domination filtering based on cost vector
        ranks, _ = do_fast_nondominated_sorting(tree_values)

        # determine internal list of efficient solutions (may include duplicates
        ax_set = [partial_trees[i] for i in range(len(tree_values)) if ranks[i] == 1]

        # remove duplicates by comparing the graphs in the efficient set
        for i in range(len(ax_set)):
            duplicate = False
            for j in range(i+1, len(ax_set)):
                if ax_set[i].is_equal_to(ax_set[j]):
                    duplicate = True
                    break
            if not duplicate:
                self.approxSet.append(ax_set[i])

        # compute objective values for the unique efficient set
        self.approxFront = [t.get_sum_of_edge_costs() for t in self.approxSet]

        return self

    def get_approximation_set(self) -> list[Graph]:
        """Get the actual efficient set."""
        return self.approxSet

    def get_approximation_front(self) -> list[list[float]]:
        """Get the objective values of the efficient set."""
        return self.approxFront
