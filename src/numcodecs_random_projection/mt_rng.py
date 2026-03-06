import concurrent.futures
import multiprocessing

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence


class MultithreadedRNG:
    """
    Multithreaded random number generator using numpy's PCG64 generator
    Based on https://numpy.org/doc/stable/reference/random/multithreading.html.
    """

    __slots__ = ("_threads", "_seed_seq", "_executor")
    _threads: int
    _seed_seq: SeedSequence
    _executor: None | concurrent.futures.ThreadPoolExecutor

    _ROWS_PER_CHUNK = 1024

    def __init__(self, seed: int, threads: None | int = None):
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
        self._threads = threads

        self._seed_seq = SeedSequence(seed)

        # only offload if there is more than one thread
        self._executor = (
            None
            if threads == 1
            else concurrent.futures.ThreadPoolExecutor(self._threads)
        )

    def fill_arr(
        self, *, out: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> None:
        """
        Fill a 2D array with random numbers in parallel using threads.

        The number of RNG chunks is determined by the shape,
        so results are reproducible regardless of the number of threads used.

        Parameters
        ----------
        out : np.ndarray[tuple[int, int], np.dtype[np.floating]]
            The 2D array to fill.
        """
        n_rows, _n_cols = out.shape

        chunk_step = self._ROWS_PER_CHUNK
        n_chunks = max(1, int(np.ceil(n_rows / chunk_step)))

        child_seeds = self._seed_seq.spawn(n_chunks)

        chunks = []
        for i in range(n_chunks):
            first = i * chunk_step
            last = min((i + 1) * chunk_step, n_rows)
            if first >= last:
                break
            chunks.append((i, first, last))

        def _fill_chunk(seed, out, first, last):
            rng = Generator(PCG64(seed))
            view = out[first:last]
            view[...] = rng.standard_normal(view.shape)

        if self._executor is None:
            for idx, first, last in chunks:
                _fill_chunk(child_seeds[idx], out, first, last)
        else:
            concurrent.futures.wait(
                self._executor.submit(_fill_chunk, child_seeds[idx], out, first, last)
                for idx, first, last in chunks
            )

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(False)
