"""Class with method for random sampling of weight vectors."""
import pyrandvec


class WeightSampler:
    """Informal interface for weight sampling."""

    def sample(self, n: int, k: int) -> list[float]:
        """Solve the problem."""
        pass


class RandomWeightSampler(WeightSampler):
    """
    Factory class for random weight vector sampling.

    This class offers a single method sample(W) which samples weights
    lambda_1, ..., lambda_W such that sum_{i=1}^{W} lambda_i = 1.
    This is realised by delegating to the standalone RandomProbabilityVectors
    module.

    Args:
        method (str): desired method (see 'method' parameter of pyrandvec.sample() function).
        shuffle (bool): shall each weight vector be randomly shuffled? Default is False.
    Returns:
        Object of type RandomWeightSampler.
    """

    def __init__(self, method = 'iterative', shuffle = False):
        """Initialise a RandomWeightSampler object."""
        self.method = method
        self.shuffle = shuffle

    def sample(self, n: int, k: int = 1) -> list[float]:
        """
        Sample a single weight vector.

        Args:
            n (int): desired dimensionality of the random weight vector.
            k (int): the number of weight vectors.
        Returns:
            A list of floating point numbers.
        """
        # just for testing purposes, but it does not hurt to keep this
        if n == 1:
            return [[1]]

        # return a list in any case -> extract the first element
        return pyrandvec.sample(k, n, self.method, self.shuffle)


class EquidistantWeightSampler(WeightSampler):
    """
    Class for sampling equidistantly distributed weights.

    This class offers a single method sample(n, k) which samples k weights
    vectors equdistantly.

    Returns:
        Object of type EquidistantWeightSampler.
    """

    def __init__(self):
        """Initialise a RandomWeightSampler object."""
        # nothing to do here

    def sample(self, n: int, k: int) -> list[list[float]]:
        """
        Sample a single weight vector.

        Args:
            n (int): desired dimensionality of the random weight vector.
            k (int): the number of vectors.
        Returns:
            list[list[float]] A list of floating point numbers.
        """
        if n != 2:
            raise ValueError('EquidistantWeightSampler works for n = 2 only.')

        return [(x / k, 1 - x / k) for x in range(k + 1)]
