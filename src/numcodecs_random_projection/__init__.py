"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec"]

from io import BytesIO
from math import ceil

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12


class RPCodec(Codec):
    def __init__(self, cr: None | float = None, k: None | int = None) -> None:
        self.cr = cr
        self.k = k
        self.R: np.ndarray | None = None

    """
    Placeholder codec that encodes data using random projection.
    """

    codec_id: str = "rp"  # type: ignore

    def _gen_R(self, D: int, K: int) -> np.ndarray:
        rng = np.random.default_rng()
        R = rng.normal(0, 1 / np.sqrt(K), size=(D, K))
        return R.astype(np.float32)

    def encode(self, buf: Buffer) -> Buffer:
        """
        Encode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded.

        Returns
        -------
        enc : bytes
            Encoded `buf`fer information as a bytestring.
        """
        a = numcodecs.compat.ensure_ndarray(buf)

        original_shape = a.shape
        original_dtype = a.dtype

        if self.k is None:
            if self.cr is not None:
                self.k = ceil(a.shape[1] / self.cr)
            else:
                raise ValueError("Parameter 'cr' must be specified for RPCodec.")

        self.R = self._gen_R(a.shape[1], self.k)
        projected = np.matmul(a, self.R)

        bio = BytesIO()

        bio.write(varint.encode(len(original_shape)))
        for dim in original_shape:
            bio.write(varint.encode(dim))

        dtype_str = original_dtype.str.encode("ascii")
        bio.write(varint.encode(len(dtype_str)))
        bio.write(dtype_str)

        bio.write(varint.encode(self.k))

        R_bytes = self.R.astype(np.float32).tobytes()
        bio.write(varint.encode(len(R_bytes)))
        bio.write(R_bytes)

        proj_bytes = projected.astype(np.float32).tobytes()
        bio.write(varint.encode(len(proj_bytes)))
        bio.write(proj_bytes)

        return bio.getvalue()

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Encoded buffer information.
        out : Buffer, optional
            Writeable buffer to store decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data.
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


numcodecs.registry.register_codec(RPCodec)
