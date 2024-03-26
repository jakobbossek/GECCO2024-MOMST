from typing import Any


class MultiObjectiveSolverInterface:
    """Interface for multi-objective optimisation algorithms / solvers."""

    def solve(self) -> 'MultiObjectiveSolverInterface':
        """Solve the problem."""
        pass

    def get_approximation_set(self) -> list[Any]:
        """Get the actual solutions."""
        pass

    def get_approximation_front(self) -> list[list[float]]:
        """Get the weight vectors of the solutions."""
        pass
