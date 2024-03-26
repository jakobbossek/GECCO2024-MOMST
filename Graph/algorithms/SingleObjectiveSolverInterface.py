from Graph.Graph import Graph


class SingleObjectiveSolverInterface:
    """Interface for single-objective optimisation algorithms / solvers."""

    def solve(self, lambdas: list[float] | None) -> 'SingleObjectiveSolverInterface':
        """Solve the problem."""
        pass

    def get_solution(self) -> Graph:
        """Get the solutions."""
        pass
