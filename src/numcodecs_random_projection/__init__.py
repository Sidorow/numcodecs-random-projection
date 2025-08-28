"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec", "RPMethod"]

import logging
import time
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from math import ceil
from sys import byteorder

import numcodecs.compat
import numcodecs.registry
import numpy as np
import tqdm
import varint
from numcodecs.abc import Codec
from numpy.random import Generator, Philox
from typing_extensions import (
    Buffer,  # MSPV 3.12
    assert_never,  # MSPV 3.11
)

LOG = logging.getLogger(__name__)


class RPMethod(Enum):
    """Random projection method."""

    dct = "dct"
    """
    Discrete Cosine Transform (DCT).

    Generate DxK projection matrix R using Type II Discrete Cosine Transform (DCT) basis ¹.
    """
    gaussian = "gaussian"
    """
    Gaussian random projection.

    Generate random DxK matrix R with entries drawn from N(0, 1/√K) distribution,
    which preserves expected distances according to Johnson-Lindenstrauss lemma.

    
    [^1]:   José J. Amador,
            Random projection and orthonormality for lossy image compression, Image and Vision Computing, Volume 25, Issue 5, 2007, Pages 754-766, ISSN 0262-8856, Available from:
            [https://doi.org/10.1016/j.imavis.2006.05.018](https://doi.org/10.1016/j.imavis.2006.05.018)
    """


class RPCodec(Codec):
    """
    Random projection codec for lossy compression of numerical data.

    Compresses 2D finite floating point data by projecting it onto a lower-dimensional subspace using a specified method.
    Discrete Cosine Transform (DCT) is used by default.

    A two-dimensional array of shape N x D is encoded as an array of
    shape N x K, where K is either set explicitly or chosen with compression ratio `cr`.
    Alternatively, K can be estimated from the data during encoding by giving a specified Mean Absolute Error (MAE) during initialization.

    """

    __slots__ = ("_mae", "_cr", "_k", "_method", "_seed", "_debug")
    _mae: None | float
    _cr: None | float
    _k: None | int
    _method: RPMethod
    _seed: int
    _debug: bool

    codec_id: str = "rp"  # type: ignore

    def __init__(
        self,
        mae: None | float = None,
        cr: None | float = None,
        k: None | int = None,
        method: str | RPMethod = RPMethod.dct,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize Random Projection codec.

        Parameters
        ----------
        mae : float
            Target mean absolute error. If specified, k will be estimated from
            data during encoding. Note that the bound is *not* guatanteed to be
            met.
        cr : float
            Target compression ratio. If specified, k will be calculated as D/cr
            where D is the number of features in the input data.
        k : int
            Number of dimensions in the projected space. Will be used over cr if
            both are specified.
        method : str | RPMethod
            Method for generating the projection matrix. Please refer to the
            [`RPMethod`][numcodecs_random_projection.RPMethod] enumeration for
            all supported methods.
        seed : int
            Random seed for reproducible results. If None, results will be
            non-deterministic when using the Gaussian method.
        debug : bool
            Whether debug information should be printed during encoding and
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

        if seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)
        else:
            self._seed = seed

        self._debug = debug

    def _estimate_k_for_target_mae(
        self,
        data: np.ndarray,
    ) -> int:
        """
        Estimate the number of dimensions k for the projected space based on the
        input data and targeted MAE.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        int
            Estimated k
        """

        D = data.shape[1]

        assert self._mae is not None
        target_mae = self._mae
        # normalized_mae = self._mae / data_std if data_std > 0 else self._mae

        match self._method:
            case RPMethod.gaussian:
                ratio = 1 - target_mae
            case RPMethod.dct:
                ratio = 1 - np.sqrt(target_mae * 4)
            case _:
                assert_never(self._method)

        estimated_k = int(D * ratio)
        K = max(1, min(estimated_k, D))

        return K

    def _project_blocks(
        self, data: np.ndarray, D: int, K: int, dtype: np.dtype, block_size: int
    ) -> np.ndarray:
        """
        Project input data to a lower-dimensional subspace using block-wise matrix generation.
        Processes projection matrix R in blocks of shape (D, block_size) instead of generating the full DxK matrix to reduce memory usage when K is large.

        Note that due to how blocks are processed and projection matrix is generated, the output is not the same as if the full matrix was used (see _gen_R_block notes).

        Parameters
        ----------
        data : np.ndarray
            Input data array with shape (N, D), where N is the number of samples
            and D is the number of input features.
        D : int
            Number of input features (columns in data).
        K : int
            Number of dimensions in the projected space.
        dtype : np.dtype
            Data type for the projection matrix and output. Should match the
            original data dtype for consistency.
        block_size : int
            Number of features to process in each block. Determines the
            size of each R_block as (D, block_size).

        Returns
        -------
        np.ndarray
            Projected data with shape (N, K)
        """

        philox = Philox(seed=self._seed)
        rng = Generator(philox)

        out = np.empty((data.shape[0], K), dtype=dtype)
        R_block = None
        out_block = None

        if self._debug:
            progress = tqdm.tqdm(total=K)

        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, K)
            actual_block_size = k_end - k_start

            if R_block is None or R_block.shape != (D, actual_block_size):
                R_block = np.empty((D, actual_block_size), dtype=dtype)

            if out_block is None or out_block.shape != (
                data.shape[0],
                actual_block_size,
            ):
                out_block = np.empty((data.shape[0], actual_block_size), dtype=dtype)

            block_timing = [0.0]
            with self._debug_timing(block_timing):
                self._gen_R_block(K, k_start, rng, out=R_block)

            matmul_timing = [0.0]
            with self._debug_timing(matmul_timing):
                np.matmul(data, R_block, out=out_block)
                out[:, k_start:k_end] = out_block

            if self._debug:
                progress.set_postfix_str(
                    f"encode N={data.shape[0]} D={D} Kb={actual_block_size} Rgen={np.round(block_timing[0], 2)}s matmul={np.round(matmul_timing[0], 2)}s"
                )
                progress.update(actual_block_size)

        return out

    @contextmanager
    def _debug_timing(self, out: list[float]):
        if self._debug:
            start = time.perf_counter()
            yield
            end = time.perf_counter()
            out[0] = end - start
        else:
            yield

    def _reconstruct_blocks(
        self,
        projected: np.ndarray,
        D: int,
        K: int,
        dtype: np.dtype,
        block_size: int,
        seed: int,
    ) -> np.ndarray:
        """
        Reconstruct data using block-wise matrix generation.

        Performs the inverse operation of _project_blocks by computing projected @ R.T in blocks to reduce memory usage. Accumulate each processed block and return the full reconstructed matrix of shape (N, D).

        Parameters
        ----------
        projected : np.ndarray
            Projected data array with shape (N, K), where N is the number of samples
            and K is the number of projected features.
        D : int
            Number of input features (columns in data).
        K : int
            Number of dimensions in the projected space.
        dtype : np.dtype
            Data type for the reconstructed matrix. Should match the
            original data dtype for consistency.
        block_size : int
            Number of features to process in each block.

        Returns
        -------
        np.ndarray
            Reconstructed data with shape (N, D)
        """

        philox = Philox(seed=seed)
        rng = Generator(philox)

        out = np.zeros((projected.shape[0], D), dtype=dtype)
        R_block = None
        rec_block = np.empty((projected.shape[0], D), dtype=dtype)

        if self._debug:
            progress = tqdm.tqdm(total=K)

        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, K)
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

            if self._debug:
                progress.set_postfix_str(
                    f"decode N={out.shape[0]} D={D} Kb={actual_block_size} Rgen={np.round(block_timing[0], 2)}s matmul={np.round(matmul_timing[0], 2)}s acc={np.round(acc_timing[0], 2)}s"
                )
                progress.update(actual_block_size)

        return out

    def _gen_R(
        self, D: int, K: int, dtype: np.dtype, seed: int | None = None
    ) -> np.ndarray:
        """
        Generate a projection matrix using specified method.

        DCT method:
            Generates a DxK projection matrix R using Type II Discrete Cosine Transform (DCT) basis.

        Gaussian method:
            Creates a random DxK matrix R with entries drawn from N(0, 1/√K) distribution,
            which preserves expected distances according to Johnson-Lindenstrauss lemma.

        Parameters
        ----------
        D : int
            Input dimensionality (number of features).
        K : int
            Output dimensionality (number of projected features).
        seed : int, optional
            Random seed of reproducible matrix generation.

        Returns
        -------
        np.ndarray
            Projection matrix of shape (D, K)
        """

        match self._method:
            case RPMethod.dct:
                i = np.arange(D, dtype=dtype).reshape(-1, 1)
                m = np.arange(K, dtype=dtype).reshape(1, -1)
                alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
                R = alpha_m * np.cos((np.pi * (2 * i + 1) * m) / (2 * D))
            case RPMethod.gaussian:
                philox = Philox(seed=seed)
                rng = Generator(philox)
                R = rng.normal(0, 1 / np.sqrt(K), size=(D, K))
            case _:
                assert_never(self._method)

        return R.astype(dtype)

    def _gen_R_block(
        self,
        K: int,
        k_start: int,
        rng: Generator,
        out: np.ndarray,
    ) -> None:
        """
        Generate a block of projection matrix R using a specified method.

        Parameters
        ----------
        K : int
            Number of projected features.
        k_start : int
            Starting index for the projected space.
        rng : Generator
            Random number generator.
        out : np.ndarray
            Block of matrix R with shape (D, block_size) that will be filled
            by this method.

        Notes
        -----
        - Generating Gaussian R matrix block by block produces the same numbers as if the full matrix was generated
            but due to the shape, the slices are not exact. If the full block generated R is concatenated by axis=0 and reshaped to (D, K),
            it would be identical to the fully generated R matrix.
        """

        D, block_size = out.shape
        dtype = out.dtype

        match self._method:
            case RPMethod.dct:
                i = np.arange(D, dtype=dtype).reshape(-1, 1)
                m = np.arange(k_start, k_start + block_size, dtype=dtype).reshape(1, -1)
                alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
                # R_block = alpha_m * np.cos((np.pi * (2 * i + 1) * m) / (2 * D))
                out[:] = 2 * i + 1
                out[:] *= m * np.pi
                out[:] /= 2 * D
                np.cos(out, out=out)
                out *= alpha_m
            case RPMethod.gaussian:
                out[:] = rng.normal(0, 1 / np.sqrt(K), size=(D, block_size))
            case _:
                assert_never(self._method)

    def encode(self, buf: Buffer) -> Buffer:
        """
        Encode data using random projection.

        Parameters
        ----------
        buf : Buffer
            Input data buffer. Must be a 2D array with shape (n_samples, d_features).

        Returns
        -------
        enc : bytes
            Serialized encoded data containing:
            - Original data shape and dtype
            - Projected data
            - Compression parameters
        """
        data = numcodecs.compat.ensure_ndarray(buf)

        validations = [
            not np.issubdtype(data.dtype, np.floating),
            data.ndim != 2,
        ]

        if any(validations):
            raise ValueError(
                f"RPCodec requires 2D floating-point data, got {data.dtype} and {data.ndim}D data"
            )

        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        data_mean = np.mean(data, axis=0, keepdims=True)
        data_std = np.std(data, axis=0, keepdims=True)
        data_std = np.where(data_std == 0, 1, data_std)

        standardized_data = (data - data_mean) / data_std

        original_shape = data.shape
        original_dtype = data.dtype

        k: int
        if self._mae is not None:
            k = self._estimate_k_for_target_mae(standardized_data)
        elif self._cr is not None:
            assert self._cr is not None
            k = int(ceil(data.shape[1] / self._cr))
        else:
            assert self._k is not None
            k = self._k

        if self._debug:
            LOG.debug(f"encode with k={k}")

        if k > 1000:
            block_size = 256
            projected = self._project_blocks(
                standardized_data, data.shape[1], k, original_dtype, block_size
            )
        else:
            R = self._gen_R(data.shape[1], k, original_dtype, self._seed)
            projected = np.matmul(standardized_data, R)

        bio = BytesIO()

        bio.write(varint.encode(len(original_shape)))
        for dim in original_shape:
            bio.write(varint.encode(dim))

        dtype_str = original_dtype.str.encode("ascii")
        bio.write(varint.encode(len(dtype_str)))
        bio.write(dtype_str)

        bio.write(varint.encode(k))
        bio.write(varint.encode(self._seed))

        mean_bytes = data_mean.astype(original_dtype).tobytes()
        bio.write(varint.encode(len(mean_bytes)))
        bio.write(mean_bytes)

        std_bytes = data_std.astype(original_dtype).tobytes()
        bio.write(varint.encode(len(std_bytes)))
        bio.write(std_bytes)

        projected_byteorder = projected.dtype.byteorder

        projected_byteorder = (
            projected_byteorder
            if projected_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )

        if projected_byteorder != "<":
            projected = projected.byteswap()

        proj_bytes = projected.tobytes()
        bio.write(varint.encode(len(proj_bytes)))
        bio.write(proj_bytes)

        return bio.getvalue()

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode random projection encoded data.

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

        ndim = varint.decode_stream(bio)
        original_shape = tuple(varint.decode_stream(bio) for _ in range(ndim))

        dtype_len = varint.decode_stream(bio)
        dtype_str = bio.read(dtype_len).decode("ascii")
        original_dtype = np.dtype(dtype_str)

        k = varint.decode_stream(bio)
        seed = varint.decode_stream(bio)

        mean_len = varint.decode_stream(bio)
        mean_bytes = bio.read(mean_len)
        data_mean = np.frombuffer(mean_bytes, dtype=original_dtype).reshape(1, -1)

        std_len = varint.decode_stream(bio)
        std_bytes = bio.read(std_len)
        data_std = np.frombuffer(std_bytes, dtype=original_dtype).reshape(1, -1)

        proj_len = varint.decode_stream(bio)
        proj_bytes = bio.read(proj_len)

        projected = np.frombuffer(
            proj_bytes, dtype=original_dtype.newbyteorder("<")
        ).reshape((original_shape[0], k))

        projected_byteorder = projected.dtype.byteorder

        projected_byteorder = (
            projected_byteorder
            if projected_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )

        if byteorder == "big":
            projected = projected.byteswap()

        if k > 1000:
            block_size = 256
            reconstructed = self._reconstruct_blocks(
                projected, original_shape[1], k, original_dtype, block_size, seed
            )
        else:
            R = self._gen_R(original_shape[1], k, original_dtype, seed)
            reconstructed = np.matmul(projected, R.T)

        reconstructed = reconstructed * data_std + data_mean
        reconstructed = reconstructed.reshape(original_shape)
        return numcodecs.compat.ndarray_copy(reconstructed, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Get codec configuration.

        Returns:
            dict: Codec configuration.
        """
        config: dict[str, str | int | float] = dict(id=type(self).codec_id)

        if self._mae is not None:
            config["mae"] = self._mae
        if self._cr is not None:
            config["cr"] = self._cr
        if self._k is not None:
            config["k"] = self._k

        config["method"] = self._method.name
        config["seed"] = self._seed

        return config


numcodecs.registry.register_codec(RPCodec)
