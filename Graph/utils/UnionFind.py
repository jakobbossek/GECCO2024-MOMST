"""Data structure for set with support for fast union and find operations."""


class UnionFind:
    """Data structure for set with support for fast union and find operations."""

    def __init__(self, n: int, sets: list[list] | None = None) -> None:
        """
        Union-Find data structure for efficient merging and search of sets.

        Args:
          n (int): Number of elements.
          sets (list of lists): Initial sets; defaults to None.
        Returns:
          Nothing.
        """
        assert n >= 2, f'number of elements expected to be greater or equal 2, but got: {n}'

        self.n: int = n

        # we add a dummy element (root[0] won't be used at all)
        self.root: list[int] = list(range(n + 1))
        self.size: list[int] = [1] * (n + 1)

        # number of sets
        self.nsets: int = n

        if sets is None:
            return

        for s in sets:
            for elem in s:
                self.root[elem] = s[0]  # w.l.o.g. the first element is the identifier
                self.size[elem] = len(s)

        self.nsets = len(sets)

    def number_of_sets(self) -> int:
        """
        Return the current number of disjoint sets.

        Returns:
          Number of sets.
        """
        return self.nsets

    def is_one_set(self) -> bool:
        """
        Check if there is only one set left.

        Returns:
          Boolean indicating whether only one set is left.
        """
        return self.nsets == 1

    def get_root(self, i: int) -> int:
        """
        Get the root element of set i.

        Side effect: path compression is used to make subsequent queries faster.

        Args:
          i (int): ID of element.
        Returns:
          ID of root element of i.
        """
        assert i >= 1 and i <= self.n

        # propagate parent
        j = i
        while j != self.root[j]:
            j = self.root[j]

        # path compression
        k = i
        while k != self.root[k]:
            tmp = k
            k = self.root[k]
            self.root[tmp] = j

        return j

    def find(self, i: int, j: int) -> bool:
        """
        Check whether two elements are in the same set.

        Args:
          i (int): ID of first element.
          j (int): ID of second element.
        Returns:
          True if both elements are in the same set and False otherwise.
        """
        return self.get_root(i) == self.get_root(j)

    def union(self, i: int, j: int) -> bool:
        """
        Merge two sets.

        Args:
          i (int): ID of first element.
          j (int): ID of second element.
        Returns:
          True if merging took place and False if i and j were already in the same set.
        """
        # optimisation: guaranteed constant time if there is just one set
        if self.nsets == 1:
            return False

        # get roots of both elements
        root_i = self.get_root(i)
        root_j = self.get_root(j)

        # nothing to do if i and j are in the same set
        if root_i == root_j:
            return False

        # otherwise append the smaller tree to the larger one
        if self.size[i] < self.size[j]:
            self.root[root_j] = i
            self.size[i] += self.size[root_j]
        else:
            self.root[root_i] = j
            self.size[j] += self.size[root_i]

        # union of two sets reduces the number of sets by one
        self.nsets -= 1

        return True
