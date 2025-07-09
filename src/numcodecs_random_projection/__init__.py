"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec"]

import warnings
from io import BytesIO
from math import ceil

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12


class RPCodec(Codec):
    """
    Random projection codec for lossy compression of numerical data.

    Compresses 2D data by projecting it onto a lower-dimensional subspace using a random Gaussian matrix.

    """

    def __init__(
        self, cr: None | float = None, k: None | int = None, seed: int | None = None
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
        seed : int, optional
            Random seed for reproducible results. If None, results will be
            non-deterministic.

        Raises
        ------
        ValueError
            If neither cr nor k is specified during encoding.
        """
        self.cr = cr
        self.k = k
        self.seed = seed

    codec_id: str = "rp"  # type: ignore

    def _gen_R(self, D: int, K: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a random projection matrix using Gaussian distribution.

        Creates a DxK matrix with entries drawn from N(0, 1/√K) distribution,
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
            Random projection matrix of shape (D, K) with dtype float32
        """
        rng = np.random.default_rng(seed)
        R = rng.normal(0, 1 / np.sqrt(K), size=(D, K))
        return R.astype(np.float32)

    def encode(self, buf: Buffer) -> Buffer:
        """
        Encode data using random projection.

        Parameters
        ----------
        buf : Buffer
            Input data buffer. Must be a 2D array with shape (n_samples, d_deatures).

        Returns
        -------
        enc : bytes
            Serialized encoded data containing:
            - Original data shape and dtype
            - Projection matrix R
            - Projected data
            - Compression parameters
        """
        a = numcodecs.compat.ensure_ndarray(buf)

        original_shape = a.shape
        original_dtype = a.dtype

        if self.k and self.cr:
            warnings.warn(
                f"Both 'cr' ({self.cr}) and 'k' ({self.k}) specified.\n Using 'k' = {self.k}",
                UserWarning,
                stacklevel=2,
            )

        elif self.k is None:
            if self.cr is not None:
                self.k = ceil(a.shape[1] / self.cr)
            else:
                raise ValueError("Parameter 'cr' or 'k' must be specified for RPCodec.")

        R = self._gen_R(a.shape[1], self.k, self.seed)
        a_32 = a.astype(np.float32)

        projected = np.matmul(a_32, R)

        bio = BytesIO()

        bio.write(varint.encode(len(original_shape)))
        for dim in original_shape:
            bio.write(varint.encode(dim))

        dtype_str = original_dtype.str.encode("ascii")
        bio.write(varint.encode(len(dtype_str)))
        bio.write(dtype_str)

        bio.write(varint.encode(self.k))

        R_bytes = R.astype(np.float32).tobytes()
        bio.write(varint.encode(len(R_bytes)))
        bio.write(R_bytes)

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
        bio = BytesIO(buf)

        ndim = varint.decode_stream(bio)
        original_shape = tuple(varint.decode_stream(bio) for _ in range(ndim))

        dtype_len = varint.decode_stream(bio)
        dtype_str = bio.read(dtype_len).decode("ascii")
        original_dtype = np.dtype(dtype_str)

        k = varint.decode_stream(bio)

        R_len = varint.decode_stream(bio)
        R_bytes = bio.read(R_len)
        R = np.frombuffer(R_bytes, dtype=np.float32).reshape((original_shape[1], k))

        proj_len = varint.decode_stream(bio)
        proj_bytes = bio.read(proj_len)
        projected = np.frombuffer(proj_bytes, dtype=np.float32).reshape(
            (original_shape[0], k)
        )

        reconstructed = np.matmul(projected, R.T)

        reconstructed = reconstructed.astype(original_dtype).reshape(original_shape)
        return numcodecs.compat.ndarray_copy(reconstructed, out)  # type: ignore

    def get_config(self) -> dict:
        return dict(id="rp", cr=self.cr, k=self.k)


numcodecs.registry.register_codec(RPCodec)
