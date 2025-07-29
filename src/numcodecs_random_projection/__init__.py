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

    def _gen_R(self, D: int, K: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a projection matrix using specified method.

        DCT method:
            Generates a projection matrix R using Discrete Cosine Transform (DCT) basis.

        Gaussian method:
            Creates a random DxK matrix R with entries drawn from N(0, 1/√K) distribution,
            which preserves expected distances according to Johnson-Lindenstrauss lemma.

        Parameters
        ----------
        D : int
            Input dimensionality (number of features)
        K : int
            Output dimensionality (number of projected features)
        seed : int, optional
            Random seed of reproducible matrix generation

        Returns
        -------
        np.ndarray
            Projection matrix of shape (D, K) with dtype float32
        """
        if self.method == "dct":

            def alpha(m):
                return np.where(m == 0, np.sqrt(1 / D), np.sqrt(2 / D))

            input_idx, output_idx = np.meshgrid(
                np.arange(D, dtype=np.float32),
                np.arange(K, dtype=np.float32),
                indexing="ij",
            )

            R = alpha(output_idx) * np.cos(
                (np.pi * (2 * input_idx + 1) * output_idx) / (2 * D)
            )

            return R.astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            R = rng.normal(0, 1 / np.sqrt(K), size=(D, K))
            return R.astype(np.float32)

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

        original_shape = data.shape
        original_dtype = data.dtype

        if self.k is None:
            if self.cr is not None:
                self.k = ceil(data.shape[1] / self.cr)

        assert self.k is not None

        R = self._gen_R(data.shape[1], self.k, self.seed)
        data_32 = data.astype(np.float32)

        projected = np.matmul(data_32, R)

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
            proj_bytes, dtype=np.dtype("f4").newbyteorder("<")
        ).reshape((original_shape[0], k))

        projected_byteorder = projected.dtype.byteorder

        projected_byteorder = (
            projected_byteorder
            if projected_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )

        if byteorder == "big":
            projected = projected.byteswap()

        R = self._gen_R(original_shape[1], k, seed)
        reconstructed = np.matmul(projected, R.T)

        reconstructed = reconstructed.astype(original_dtype).reshape(original_shape)
        return numcodecs.compat.ndarray_copy(reconstructed, out)  # type: ignore

    def get_config(self) -> dict:
        return dict(id="rp", cr=self.cr, k=self.k, seed=self.seed)


numcodecs.registry.register_codec(RPCodec)
