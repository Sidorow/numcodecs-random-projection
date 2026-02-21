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

        self.seed_seq = SeedSequence(seed)
        self.shape = None
        self.values = None
        self.step = None

        self.executor = concurrent.futures.ThreadPoolExecutor(self.threads)

    def fill_arr(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Fill an array of given shape with random numbers in parallel using threads.

        The number of RNG chunks is determined by the shape,
        so results are reproducible regardless of the number of threads used.

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
        n_rows = self.values.shape[0]

        rows_per_chunk = shape[1]
        n_chunks = max(1, int(np.ceil(n_rows / rows_per_chunk)))
        chunk_step = int(np.ceil(n_rows / n_chunks))

        child_seeds = self.seed_seq.spawn(n_chunks)

        chunks = []
        for i in range(n_chunks):
            first = i * chunk_step
            last = min((i + 1) * chunk_step, n_rows)
            if first >= last:
                break
            chunks.append((i, first, last))

        def _fill_chunk(chunk_idx, out, first, last):
            rng = default_rng(child_seeds[chunk_idx])
            view = out[first:last]
            view[...] = rng.standard_normal(view.shape)

        futures = [
            self.executor.submit(_fill_chunk, idx, self.values, first, last)
            for idx, first, last in chunks
        ]

        concurrent.futures.wait(futures)

        return self.values

    def __del__(self):
        self.executor.shutdown(False)
