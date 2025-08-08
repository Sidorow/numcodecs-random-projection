"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec"]

import warnings
from io import BytesIO
from math import ceil
from sys import byteorder

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from numpy.random import Generator, Philox
from typing_extensions import Buffer  # MSPV 3.12


class RPCodec(Codec):
    """
    Random projection codec for lossy compression of numerical data.

    Compresses 2D data by projecting it onto a lower-dimensional subspace using a specified method.
    Discrete Cosine Transform (DCT) is used by default.

    """

    def __init__(
        self,
        cr: None | float = None,
        k: None | int = None,
        method: str = "dct",
        seed: int | None = None,
    ) -> None:
        """
        Initialize Random Projection codec.

        Parameters
        ----------
        cr : float, optional
            Target compression ratio. If specified, k will be calculated as D/cr
            where D is the number of features in the input data.
        k : int, optional
            Number of dimensions in the projected space. Will be used over cr if
            both are specified.
        method : str, default "dct"
            Method for generating the projection matrix. Supported methods:
            - "dct": Uses Discrete Cosine Transform basis.
            - "gaussian": Uses Gaussian random projection.
        seed : int, optional
            Random seed for reproducible results. If None, results will be
            non-deterministic when using the Gaussian method.

        Raises
        ------
        ValueError
            If neither cr nor k is specified during encoding.
        """
        self.cr = cr
        self.k = k
        self.method = method

        if self.method not in ["dct", "gaussian"]:
            raise ValueError(
                f"Unknown method '{self.method}'. Supported methods: 'dct', 'gaussian'."
            )

        if seed is None:
            self.seed = np.random.randint(0, 2**31 - 1)
        else:
            self.seed = seed

        if self.k and self.cr:
            warnings.warn(
                f"Both 'cr' ({self.cr}) and 'k' ({self.k}) specified.\n Using 'k' = {self.k}",
                UserWarning,
                stacklevel=2,
            )

        elif self.k is None and self.cr is None:
            raise ValueError("Parameter 'cr' or 'k' must be specified for RPCodec.")

    codec_id: str = "rp"  # type: ignore

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
        projected_blocks = []
        philox = Philox(seed=self.seed)
        rng = Generator(philox)
        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, K)
            actual_block_size = k_end - k_start

            R_block = self._gen_R_block(D, K, k_start, dtype, actual_block_size, rng)
            print(f"R_block:\n{R_block}")

            block_proj = np.matmul(data, R_block)
            projected_blocks.append(block_proj)

            del R_block

        projected = np.concatenate(projected_blocks, axis=1).astype(dtype)
        return projected

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
        reconstructed_blocks = np.zeros((projected.shape[0], D), dtype=dtype)
        philox = Philox(seed=seed)
        rng = Generator(philox)
        for k_start in range(0, K, block_size):
            k_end = min(k_start + block_size, K)
            actual_block_size = k_end - k_start

            R_block = self._gen_R_block(D, K, k_start, dtype, actual_block_size, rng)
            R_block_T = R_block.T
            del R_block

            projected_block = projected[:, k_start:k_end]
            rec_block = np.matmul(projected_block, R_block_T)

            reconstructed_blocks += rec_block
            del R_block_T, rec_block

        return reconstructed_blocks

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
        if self.method == "dct":
            i = np.arange(D, dtype=dtype).reshape(-1, 1)
            m = np.arange(K, dtype=dtype).reshape(1, -1)
            alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
            R = alpha_m * np.cos((np.pi * (2 * i + 1) * m) / (2 * D))

        else:
            philox = Philox(seed=seed)
            rng = Generator(philox)
            R = rng.normal(0, 1 / np.sqrt(K), size=(D, K))

        return R.astype(dtype)

    def _gen_R_block(
        self,
        D: int,
        K: int,
        k_start: int,
        dtype: np.dtype,
        block_size: int,
        rng: Generator,
    ) -> np.ndarray:
        """
        Generate a block of projection matrix R using a specified method.

        Parameters
        ----------
        D : int
            Number of input features.
        k_start : int
            Starting index for the projected space.
        dtype : np.dtype
            Data type for the matrix. Should match the
            original data dtype for consistency.
        block_size : int
            Size of the block to generate.

        Returns
        -------
        np.ndarray
            Block of matrix R with shape (D, block_size)

        Notes
        -----
        - Generating Gaussian R matrix block by block produces the same numbers as if the full matrix was generated
            but due to the shape, the slices are not exact. If the full block generated R is concatenated by axis=0 and reshaped to (D, K),
            it would be identical to the fully generated R matrix.
        """
        if self.method == "dct":
            i = np.arange(D, dtype=dtype).reshape(-1, 1)
            m = np.arange(k_start, k_start + block_size, dtype=dtype).reshape(1, -1)
            alpha_m = np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))
            R_block = alpha_m * np.cos((np.pi * (2 * i + 1) * m) / (2 * D))

        else:
            R_block = rng.normal(0, 1 / np.sqrt(K), size=(D, block_size)).astype(dtype)

        return R_block

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

        original_shape = data.shape
        original_dtype = data.dtype

        if self.k is None:
            if self.cr is not None:
                self.k = ceil(data.shape[1] / self.cr)

        assert self.k is not None

        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.k > 1000:
            block_size = 500  # Arbitrary for now. Maybe calculate optimal later?
            projected = self._project_blocks(
                data, data.shape[1], self.k, original_dtype, block_size
            )
        else:
            R = self._gen_R(data.shape[1], self.k, original_dtype, self.seed)
            projected = np.matmul(data, R)

        bio = BytesIO()

        bio.write(varint.encode(len(original_shape)))
        for dim in original_shape:
            bio.write(varint.encode(dim))

        dtype_str = original_dtype.str.encode("ascii")
        bio.write(varint.encode(len(dtype_str)))
        bio.write(dtype_str)

        bio.write(varint.encode(self.k))
        bio.write(varint.encode(self.seed))

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
            block_size = 500
            reconstructed = self._reconstruct_blocks(
                projected, original_shape[1], k, original_dtype, block_size, seed
            )
        else:
            R = self._gen_R(original_shape[1], k, original_dtype, seed)
            reconstructed = np.matmul(projected, R.T)

        reconstructed = reconstructed.reshape(original_shape)
        return numcodecs.compat.ndarray_copy(reconstructed, out)  # type: ignore

    def get_config(self) -> dict:
        return dict(id="rp", cr=self.cr, k=self.k, seed=self.seed)


numcodecs.registry.register_codec(RPCodec)
