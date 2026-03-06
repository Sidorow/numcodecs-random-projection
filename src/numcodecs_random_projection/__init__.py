"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec", "RPMethod"]

import logging
import time
from contextlib import contextmanager
from enum import Enum
from functools import reduce
from io import BytesIO
from math import ceil
from typing import TypeVar

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
import tqdm
from numcodecs.abc import Codec
from typing_extensions import (
    Buffer,  # MSPV 3.12
    Unpack,  # MSPV 3.11
    assert_never,  # MSPV 3.11
)

from .mt_rng import MultithreadedRNG

LOG = logging.getLogger(__name__)


Nc = TypeVar("Nc", bound=int, covariant=True)
""" The number of samples / rows (covariant). """

Dc = TypeVar("Dc", bound=int, covariant=True)
""" The number of features / columns (covariant). """

Di = TypeVar("Di", bound=int)
""" The number of features / columns (invariant). """

Kc = TypeVar("Kc", bound=int, covariant=True)
""" The number of projected features / columns (covariant). """

Ki = TypeVar("Ki", bound=int)
""" The number of projected features / columns (invariant). """

Fc = TypeVar("Fc", bound=np.floating, covariant=True)
""" Any numpy [`floating`][numpy.floating]-point data type (covariant). """


class RPMethod(Enum):
    """Random projection method."""

    dct = "dct"
    """
    Discrete Cosine Transform (DCT).

    Generate DxK projection matrix R using Type II Discrete Cosine Transform
    (DCT) basis [^1].

    [^1]: Amador, J. J. (2006). Random projection and orthonormality for lossy
          image compression. *Image and Vision Computing*, 25(5), 754–766.
          Available from:
          [doi:10.1016/j.imavis.2006.05.018](https://doi.org/10.1016/j.imavis.2006.05.018).
    """

    gaussian = "gaussian"
    """
    Gaussian random projection.

    Generate random DxK matrix R with entries drawn from `N(0, 1/sqrt(K))`
    distribution, which preserves expected distances according to
    Johnson-Lindenstrauss lemma.
    """


class RPCodec(Codec):
    """
    Random projection codec for lossy compression of numerical data.

    Compresses 2D finite floating point data by projecting it onto a
    lower-dimensional subspace using a specified method. The Discrete Cosine
    Transform (DCT) is used by default.

    A two-dimensional array of shape N x D is encoded as an array of
    shape N x K, where `k` is either set explicitly or chosen based on the
    compression ratio `cr`. Alternatively, `k` can be estimated from the data
    during encoding by giving a loose Mean Absolute Error (MAE) bound.

    Arrays that are not two-dimensional are automatically reshaped to be 2D as
    follows:
    - 0D scalar -> 1x1
    - 1D array of shape D -> 1xD
    - 2D array of shape NxD -> NxD
    - >2D arrays: the dimensions are automatically partitioned into two subsets
      ...N and ...D that balance the product dimensions NxD; to use a different
      partitioning, you need to manually transpose and reshape into 2D before
      encoding with this codec
    """

    __slots__ = ("_mae", "_cr", "_k", "_method", "_seed", "_debug")
    _mae: None | float
    _cr: None | float
    _k: None | int
    _method: RPMethod
    _seed: None | int
    _max_block_memory: None | int
    _debug: bool

    codec_id: str = "rp"  # type: ignore

    def __init__(
        self,
        mae: None | float = None,
        cr: None | float = None,
        k: None | int = None,
        method: str | RPMethod = RPMethod.dct,
        seed: None | int = None,
        max_block_memory: None | int = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize Random Projection codec.

        Parameters
        ----------
        mae : None | float
            Target mean absolute error. If specified, `k` will be estimated from
            data during encoding. Note that the bound is *not* guaranteed to be
            met.
        cr : None | float
            Target compression ratio. If specified, `k` will be calculated as
            `D/cr` where `D` is the number of features in the input data.
        k : None | int
            Number of dimensions in the projected space. Will be used over `cr`
            if both are specified. Estimated if `mae` is specified.
        method : str | RPMethod
            Method for generating the projection matrix. Please refer to the
            [`RPMethod`][numcodecs_random_projection.RPMethod] enumeration for
            all supported methods.
        seed : None | int
            Random seed for reproducible results. If [`None`], the seed is
            determined non-deterministically at encoding time.
        max_block_memory : None | int
            Maximum non-negative amount of memory, in bytes, that a projection
            matrix block should not exceed. If small or zero, the blocks will
            be as small as possible. If `-1`, the projection matrix is produced
            in one block, no matter how large. If `None`, the available amount
            of memory is determined non-deterministically at encoding time.
        debug : bool
            Whether debug information should be logged during encoding and
            decoding.

        Raises
        ------
        ValueError
            If not exactly one of `mae`, `cr`, or `k` is set.
        """

        if sum([(mae is not None), (cr is not None), (k is not None)]) != 1:
            raise ValueError("exactly one of `mae`, `cr` or `k` must be set")

        self._mae = mae
        self._cr = cr
        self._k = k

        try:
            self._method = method if isinstance(method, RPMethod) else RPMethod[method]
        except KeyError:
            hy = "'"
            raise ValueError(
                f"unknown method '{method}', expected one of {', '.join(f'{hy}{m.name}{hy}' for m in RPMethod)}."
            )

        self._seed = seed

        if (max_block_memory is not None) and (max_block_memory < -1):
            raise ValueError("max_block_memory should be non-negative or -1")
        self._max_block_memory = max_block_memory

        self._debug = debug

    def encode(self, buf: Buffer) -> Buffer:  # type: ignore
        """
        Encode data using random projection.

        During encode, the input data is standardized (mean=0, std=1) before
        projection.

        If `mae` is specified, the number of projected dimensions `k` is
        estimated based on the standardized data.

        Parameters
        ----------
        buf : Buffer
            Input data buffer. Must be an n-dimensional floating-point array.

        Returns
        -------
        enc : bytes
            Serialized encoded data containing:
            - Standardized data statistics (mean, std)
            - Original data shape and dtype
            - Projected data
            - Compression parameters
        """
        data = np.copy(numcodecs.compat.ensure_ndarray(buf))

        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(
                f"RPCodec requires floating-point data, got {data.dtype} data"
            )

        np.nan_to_num(data, copy=False, nan=0, posinf=0, neginf=0)

        # store the data shape, dtype, mean, and std before any processing
        shape = data.shape
        dtype = data.dtype

        data_mean = np.mean(data)
        data_std = np.std(data)
        if data_std == 0:
            data_std = dtype.type(1)
        data -= data_mean
        data /= data_std

        ias: tuple[int, ...]
        ibs: tuple[int, ...]
        if len(shape) == 0:
            ias, ibs = (), ()  # 1x1
        elif len(shape) == 1:
            ias, ibs = (), (0,)  # 1xD
        elif len(shape) == 2:
            ias, ibs = (0,), (1,)  # NxD
        else:
            ias, ibs = (  # (...N)x(...D)
                self._find_balanced_shape_partition(shape)  # type: ignore
            )
        N: int = reduce(lambda acc, ia: acc * shape[ia], ias, 1)
        D: int = reduce(lambda acc, ib: acc * shape[ib], ibs, 1)
        # transpose the shape to (...N, ...D) and reshape to (N, D)
        data = np.transpose(data, axes=(*ias, *ibs)).reshape(N, D)

        if self._debug:
            LOG.debug(
                f"reshaped {shape} into N={tuple(shape[ia] for ia in ias)}={N} x D={tuple(shape[ib] for ib in ibs)}={D} with {ias},{ibs}"
            )

        standardised_matrix: np.ndarray[tuple[int, int], np.dtype[np.floating]] = data

        k: int
        if self._mae is not None:
            k = self._estimate_k_for_target_mae(standardised_matrix)
        elif self._cr is not None:
            assert self._cr is not None
            k = int(ceil(D / self._cr))
        else:
            assert self._k is not None
            k = self._k

        seed: int
        if self._seed is None:
            if self._method == RPMethod.dct:
                # no random seed is needed, keep the output fully reproducible
                seed = 0
            else:
                seed = np.random.randint(0, 2**31 - 1)
        else:
            seed = self._seed

        block_size: int
        if self._max_block_memory is None:
            block_size = self._compute_block_size(D, dtype, max_memory=None)
        elif self._max_block_memory == -1:
            block_size = k
        else:
            block_size = self._compute_block_size(
                D, dtype, max_memory=self._max_block_memory
            )

        if self._debug:
            LOG.debug(f"encode with k={k} and block_size={block_size}")

        projected: np.ndarray[tuple[int, int], np.dtype[np.floating]]
        if k > (block_size * 2):
            projected = self._project_blocks(standardised_matrix, k, seed, block_size)
        else:
            R = self._gen_R(D, k, dtype, seed)
            projected = np.matmul(standardised_matrix, R)

        # shape ias ibs dtype K block_size seed mean std projected
        bio = BytesIO()

        bio.write(leb128.u.encode(len(shape)))
        for dim in shape:
            bio.write(leb128.u.encode(dim))

        bio.write(leb128.u.encode(len(ias)))
        for ia in ias:
            bio.write(leb128.u.encode(ia))
        bio.write(leb128.u.encode(len(ibs)))
        for ib in ibs:
            bio.write(leb128.u.encode(ib))

        dtype_str = dtype.str.encode("ascii")
        bio.write(leb128.u.encode(len(dtype_str)))
        bio.write(dtype_str)

        bio.write(leb128.u.encode(k))
        bio.write(leb128.u.encode(block_size))
        bio.write(leb128.u.encode(seed))

        data_mean = np.array(data_mean, dtype=dtype)
        mean_bytes = data_mean.astype(data_mean.dtype.newbyteorder("<")).tobytes()
        bio.write(leb128.u.encode(len(mean_bytes)))
        bio.write(mean_bytes)

        data_std = np.array(data_std, dtype=dtype)
        std_bytes = data_std.astype(data_std.dtype.newbyteorder("<")).tobytes()
        bio.write(leb128.u.encode(len(std_bytes)))
        bio.write(std_bytes)

        proj_bytes = projected.astype(projected.dtype.newbyteorder("<")).tobytes()
        bio.write(leb128.u.encode(len(proj_bytes)))
        bio.write(proj_bytes)

        return bio.getvalue()

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:  # type: ignore
        """
        Decode random projection encoded data.

        During decode, the standardized data is reconstructed and denormalized.

        Parameters
        ----------
        buf : Buffer
            Encoded data from RPCodec.
        out : Buffer, optional
            Writeable buffer to store decoded data.

        Returns
        -------
        dec : Buffer
            Reconstructed data with original shape and dtype.
        """

        data = numcodecs.compat.ensure_bytes(buf)

        bio = BytesIO(data)

        ndim, _ = leb128.u.decode_reader(bio)
        shape = tuple(leb128.u.decode_reader(bio)[0] for _ in range(ndim))

        ias = tuple(
            leb128.u.decode_reader(bio)[0]
            for _ in range(leb128.u.decode_reader(bio)[0])
        )
        ibs = tuple(
            leb128.u.decode_reader(bio)[0]
            for _ in range(leb128.u.decode_reader(bio)[0])
        )
        N: int = reduce(lambda acc, ia: acc * shape[ia], ias, 1)
        D: int = reduce(lambda acc, ib: acc * shape[ib], ibs, 1)

        dtype_len, _ = leb128.u.decode_reader(bio)
        dtype_str = bio.read(dtype_len).decode("ascii")
        dtype = np.dtype(dtype_str)

        k, _ = leb128.u.decode_reader(bio)
        block_size, _ = leb128.u.decode_reader(bio)
        seed, _ = leb128.u.decode_reader(bio)

        mean_len, _ = leb128.u.decode_reader(bio)
        mean_bytes = bio.read(mean_len)
        data_mean = np.frombuffer(
            mean_bytes, dtype=dtype.newbyteorder("<"), count=1
        ).astype(dtype)[0]

        std_len, _ = leb128.u.decode_reader(bio)
        std_bytes = bio.read(std_len)
        data_std = np.frombuffer(
            std_bytes, dtype=dtype.newbyteorder("<"), count=1
        ).astype(dtype)[0]

        proj_len, _ = leb128.u.decode_reader(bio)
        proj_bytes = bio.read(proj_len)

        projected = (
            np.frombuffer(
                proj_bytes,
                dtype=dtype.newbyteorder("<"),
                count=(N * k),
            )
            .astype(dtype)
            .reshape((N, k))
        )

        reconstructed: np.ndarray[tuple[int, int], np.dtype[np.floating]]
        if k > (block_size * 2):
            reconstructed = self._reconstruct_blocks(projected, D, block_size, seed)
        else:
            R = self._gen_R(D, k, dtype, seed)
            reconstructed = np.matmul(projected, R.T)

        reconstructed *= data_std
        reconstructed += data_mean

        # split reconstructed from (N, D) into (...N, ...D)
        reconstructed_: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            reconstructed.reshape(
                tuple(shape[ia] for ia in ias) + tuple(shape[ib] for ib in ibs)
            )
        )
        # undo the axis tranpose to get from (...N, ...D) to shape
        iabs = (*ias, *ibs)
        iabs_inv = tuple(np.argsort(iabs))
        assert [iabs[i] for i in iabs_inv] == list(range(len(shape)))
        reconstructed_ = np.transpose(reconstructed_, axes=iabs_inv)

        assert reconstructed_.shape == shape

        return numcodecs.compat.ndarray_copy(reconstructed_, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Get codec configuration.

        Returns:
            dict: Codec configuration.
        """
        config: dict[str, str | int | float | bool] = dict(id=type(self).codec_id)

        if self._mae is not None:
            config["mae"] = self._mae
        if self._cr is not None:
            config["cr"] = self._cr
        if self._k is not None:
            config["k"] = self._k

        config["method"] = self._method.name

        if self._seed is not None:
            config["seed"] = self._seed
        if self._max_block_memory is not None:
            config["max_block_memory"] = self._max_block_memory
        config["debug"] = self._debug

        return config

    def _estimate_k_for_target_mae(
        self,
        data: np.ndarray[tuple[Nc, Dc], np.dtype[Fc]],
    ) -> int:
        """
        Estimate the number of dimensions 'k' for the projected space based on
        the standardized input data and targeted MAE.

        This method assumes standardized input data prior to calling via encode
        method.

        Parameters
        ----------
        data : np.ndarray[tuple[Nc, Dc], np.dtype[Fc]]
            Standardized input data (mean=0, std=1).

        Returns
        -------
        int
            Estimated k (number of projected dimensions)
        """

        _N, D = data.shape

        assert self._mae is not None
        target_mae = self._mae

        match self._method:
            case RPMethod.gaussian:
                ratio = 1 - target_mae
            case RPMethod.dct:
                ratio = 1 - np.sqrt(target_mae * 4)
            case _:
                assert_never(self._method)

        estimated_k = int(D * ratio)
        K = max(1, min(estimated_k, int(D)))

        return K

    def _project_blocks(
        self,
        data: np.ndarray[tuple[Nc, Dc], np.dtype[Fc]],
        K: Ki,
        seed: int,
        block_size: int,
    ) -> np.ndarray[tuple[Nc, Ki], np.dtype[Fc]]:
        """
        Project input data to a lower-dimensional subspace using block-wise
        matrix generation. Processes projection matrix R in blocks of shape
        (D, block_size) instead of generating the full DxK matrix to reduce
        memory usage when K is large.

        Note that due to how blocks are processed and projection matrix is
        generated, the output is not the same as if the full matrix was used
        (see _gen_R_block notes).

        Parameters
        ----------
        data : np.ndarray[tuple[Nc, Dc], np.dtype[Fc]]
            Input data array with shape (N, D), where N is the number of samples
            and D is the number of input features.
        K : Ki
            Number of dimensions in the projected space.
        seed : int
            Random seed of reproducible matrix generation.
        block_size : int
            Number of features to process in each block. Determines the
            size of each R_block as (D, block_size).

        Returns
        -------
        np.ndarray[tuple[Nc, Ki], np.dtype[Fc]]
            Projected data with shape (N, K)
        """

        rng = MultithreadedRNG(seed=seed)

        N, D = data.shape
        dtype = data.dtype

        out: np.ndarray[tuple[Nc, Ki], np.dtype[Fc]] = np.empty((N, K), dtype=dtype)
        R_block: None | np.ndarray[tuple[Dc, int], np.dtype[Fc]] = None
        out_block: None | np.ndarray[tuple[Nc, int], np.dtype[Fc]] = None

        if self._debug:
            progress = tqdm.tqdm(total=K)
        else:
            progress = None

        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, int(K))
            actual_block_size = k_end - k_start

            if R_block is None or R_block.shape != (D, actual_block_size):
                R_block = np.empty((D, actual_block_size), dtype=dtype)

            if out_block is None or out_block.shape != (
                N,
                actual_block_size,
            ):
                out_block = np.empty((N, actual_block_size), dtype=dtype)

            block_timing = [0.0]
            with self._debug_timing(block_timing):
                self._gen_R_block(K, k_start, rng, out=R_block)

            matmul_timing = [0.0]
            with self._debug_timing(matmul_timing):
                np.matmul(data, R_block, out=out_block)
                out[:, k_start:k_end] = out_block

            if progress is not None:
                progress.set_postfix_str(
                    f"encode N={N} D={D} Kb={actual_block_size} Rgen={np.round(block_timing[0], 2)}s matmul={np.round(matmul_timing[0], 2)}s"
                )
                progress.update(actual_block_size)

        return out

    def _reconstruct_blocks(
        self,
        projected: np.ndarray[tuple[Nc, Kc], np.dtype[Fc]],
        D: Di,
        block_size: int,
        seed: int,
    ) -> np.ndarray[tuple[Nc, Di], np.dtype[Fc]]:
        """
        Reconstruct data using block-wise matrix generation.

        Performs the inverse operation of _project_blocks by computing
        projected @ R.T in blocks to reduce memory usage. Accumulate each
        processed block and return the full reconstructed matrix of shape
        (N, D).

        Parameters
        ----------
        projected : np.ndarray[tuple[Nc, Kc], np.dtype[Fc]]
            Projected data array with shape (N, K), where N is the number of
            samples and K is the number of projected features.
        D : Di
            Number of input features (columns in data).
        block_size : int
            Number of features to process in each block.
        seed : int
            Random seed of reproducible matrix generation.

        Returns
        -------
        np.ndarray[tuple[Nc, Di], np.dtype[Fc]]
            Reconstructed data with shape (N, D)
        """

        rng = MultithreadedRNG(seed=seed)

        N, K = projected.shape
        dtype = projected.dtype

        out: np.ndarray[tuple[Nc, Di], np.dtype[Fc]] = np.zeros((N, D), dtype=dtype)
        R_block: None | np.ndarray[tuple[Di, int], np.dtype[Fc]] = None
        rec_block: np.ndarray[tuple[Nc, Di], np.dtype[Fc]] = np.empty(
            (N, D), dtype=dtype
        )

        if self._debug:
            progress = tqdm.tqdm(total=K)
        else:
            progress = None

        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, int(K))
            actual_block_size = k_end - k_start

            if R_block is None or R_block.shape != (D, actual_block_size):
                R_block = np.empty((D, actual_block_size), dtype=dtype)

            block_timing = [0.0]
            with self._debug_timing(block_timing):
                self._gen_R_block(K, k_start, rng, out=R_block)

            matmul_timing = [0.0]
            with self._debug_timing(matmul_timing):
                projected_block = projected[:, k_start:k_end]
                np.matmul(projected_block, R_block.T, out=rec_block)

            acc_timing = [0.0]
            with self._debug_timing(acc_timing):
                out += rec_block

            if progress is not None:
                progress.set_postfix_str(
                    f"decode N={N} D={D} Kb={actual_block_size} Rgen={np.round(block_timing[0], 2)}s matmul={np.round(matmul_timing[0], 2)}s acc={np.round(acc_timing[0], 2)}s"
                )
                progress.update(actual_block_size)

        return out

    def _compute_block_size(
        self, D: int, dtype: np.dtype, max_memory: None | int
    ) -> int:
        if max_memory is None:
            try:
                import psutil

                available = psutil.virtual_memory().available
            except ImportError:
                if self._debug:
                    LOG.warning(
                        "Cannot import psutil, using 2 GiB fallback for available memory."
                    )
                available = 2**31

            available = available // 5
        else:
            available = max_memory

        block_size = max(1, available // (D * dtype.itemsize))

        if self._debug:
            LOG.debug(
                f"Available memory: {available / (1024**2):.2f} MiB, block size: {block_size}"
            )

        return block_size

    def _gen_R(
        self, D: Di, K: Ki, dtype: np.dtype[Fc], seed: int
    ) -> np.ndarray[tuple[Di, Ki], np.dtype[Fc]]:
        """
        Generate a projection matrix using specified method.

        DCT method:
            Generates a DxK projection matrix R using Type II Discrete Cosine
            Transform (DCT) basis.

        Gaussian method:
            Creates a random DxK matrix R with entries drawn from
            N(0, 1/sqrt(K)) distribution, which preserves expected distances
            according to Johnson-Lindenstrauss lemma.

        Parameters
        ----------
        D : Di
            Input dimensionality (number of features).
        K : Ki
            Output dimensionality (number of projected features).
        dtype : np.dtype[Fc]
            Output dtype.
        seed : int
            Random seed of reproducible matrix generation.

        Returns
        -------
        np.ndarray[tuple[Di, Ki], np.dtype[Fc]]
            Projection matrix of shape (D, K)
        """

        match self._method:
            case RPMethod.dct:
                i = np.arange(D, dtype=dtype).reshape(-1, 1)
                m = np.arange(K, dtype=dtype).reshape(1, -1)
                alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
                R = alpha_m * np.cos((np.pi * (2 * i + 1) * m) / (2 * D))
            case RPMethod.gaussian:
                scale = np.sqrt(1 / K)
                rng = MultithreadedRNG(seed=seed)
                R = np.empty((D, K), dtype=dtype)
                rng.fill_arr(out=R)
                R *= scale
            case _:
                assert_never(self._method)

        return R.astype(dtype)

    def _gen_R_block(
        self,
        K: int,
        k_start: int,
        rng: MultithreadedRNG,
        out: np.ndarray[tuple[Dc, int], np.dtype[Fc]],
    ) -> None:
        """
        Generate a block of projection matrix R using a specified method.

        Parameters
        ----------
        K : int
            Number of projected features.
        k_start : int
            Starting index for the projected space.
        rng : MultithreadedRNG
            Random number generator based on
            <https://numpy.org/doc/stable/reference/random/multithreading.html>.
        out : np.ndarray[tuple[Dc, int], np.dtype[Fc]]
            Block of matrix R with shape (D, block_size) that will be filled
            by this method.

        Notes
        -----
        - Generating Gaussian R matrix block by block does not produce the same
          numbers as the Gen_R method. This is because the random numbers are
          generated in chunks based on the block size, so the sequence of random
          numbers used for each block is different than if the full matrix was
          generated at once.
        """

        D, block_size = out.shape
        dtype = out.dtype

        match self._method:
            case RPMethod.dct:
                i = np.arange(D, dtype=dtype).reshape(-1, 1)
                m = np.arange(k_start, k_start + block_size, dtype=dtype).reshape(1, -1)
                alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
                out[:] = 2 * i + 1
                out[:] *= m * np.pi
                out[:] /= 2 * D
                np.cos(out, out=out)
                out *= alpha_m
            case RPMethod.gaussian:
                scale = np.sqrt(1 / K)
                rng.fill_arr(out=out)
                out *= scale
            case _:
                assert_never(self._method)

    # adapted from https://bj0z.wordpress.com/2011/03/07/the-balanced-partition-problem/
    def _find_balanced_shape_partition(
        self, shape: tuple[int, int, Unpack[tuple[int, ...]]]
    ) -> tuple[
        tuple[int, Unpack[tuple[int, ...]]], tuple[int, Unpack[tuple[int, ...]]]
    ]:
        # simplify by treating empty dimensions as if they have size 1
        # precompute log(x) and optimise the partition sum (instead of product)
        log_shape: tuple[float, ...] = tuple(np.log(max(x, 1)) for x in shape)

        best_cost: float = sum(log_shape)
        best_subset: tuple[int, ...] = ()

        target_cost = best_cost / 2

        # memoisation dictionary
        P: dict[float, dict[float, tuple[int, ...]]] = {-1: {0: ()}}

        for i, x in enumerate(log_shape):
            P[i] = {}
            for j in P[i - 1].keys():
                P[i][j] = P[i - 1][j]
                if (j + x) > target_cost:
                    continue
                P[i][j + x] = P[i - 1][j] + (i,)

                if abs((j + x) - target_cost) < abs(best_cost - target_cost):
                    best_cost = j + x
                    best_subset = P[i][j + x]

                    if best_cost == target_cost:
                        break
            if best_cost == target_cost:
                break

        # sort the dimension indices to switch up the transpose order as little
        #  as possible
        suba: tuple[int, ...] = tuple(sorted(best_subset))
        subb: tuple[int, ...] = tuple(sorted(set(range(len(shape))) - set(suba)))

        assert len(suba) >= 1
        assert len(subb) >= 1
        assert sorted((*suba, *subb)) == list(range(len(shape)))

        return suba, subb  # type: ignore

    @contextmanager
    def _debug_timing(self, out: list[float]):
        if self._debug:
            start = time.perf_counter()
            yield
            end = time.perf_counter()
            out[0] = end - start
        else:
            yield


numcodecs.registry.register_codec(RPCodec)
