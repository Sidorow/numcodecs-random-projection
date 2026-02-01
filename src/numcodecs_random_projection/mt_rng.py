import concurrent.futures
import multiprocessing

import numpy as np
from numpy.random import SeedSequence, default_rng


class MultithreadedRNG:
    """
    Multithreaded random number generator using numpy's default_rng
    Based on https://numpy.org/doc/stable/reference/random/multithreading.html.
    """

    def __init__(self, seed, threads=None):
        """
        Initialize the multithreaded RNG.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        threads : int, optional
            The number of threads to use. If None, uses the number of CPU cores.
        """
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(threads)]
        self.shape = None
        self.values = None
        self.step = None

        self.executor = concurrent.futures.ThreadPoolExecutor(self.threads)

    def fill_arr(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Fill an array of given shape with random numbers in parallel using threads.

        Parameters
        ----------
        shape : tuple[int]
            The shape of the array to fill.

        Returns
        -------
        np.ndarray
            The filled array with random floating-point numbers.
        """
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.values = np.empty(self.shape)

        self._part_len = int(self.values.shape[0])
        self.step = int(np.ceil(self._part_len / self.threads))

        def _fill(random_state, out, first, last):
            last = min(last, out.shape[0])
            if first >= last:
                return
            view = out[first:last]
            view[...] = random_state.standard_normal(view.shape)

        futures = []
        for i in range(self.threads):
            first = i * self.step
            last = (i + 1) * self.step
            fut = self.executor.submit(
                _fill, self._random_generators[i], self.values, first, last
            )
            futures.append(fut)

        concurrent.futures.wait(futures)

        return self.values

    def __del__(self):
        self.executor.shutdown(False)
