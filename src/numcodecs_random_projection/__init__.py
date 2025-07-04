"""
[`RPCodec`][numcodecs_random_projection.RPCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["RPCodec"]

from io import BytesIO

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from math import ceil
from typing_extensions import Buffer  # MSPV 3.12


class RPCodec(Codec):
    def __init__(self, cr: None | float = None, k: None | int = None) -> None:
        self.cr = cr
        self.k = k
        self.R = None
        if cr is None and k is None:
            raise ValueError("Parameters 'cr' or 'k' must be specified for RPCodec.")
        
    """
    Placeholder codec that encodes data using random projection.
    """

    codec_id: str = "rp"  # type: ignore

    def _gen_R(self, D, K):
        rng = np.random.default_rng()
        R = rng.normal(0, 1/np.sqrt(K), size=(D, K))
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
        
        if self.k is None and self.cr is not None:
            self.k = ceil(buf.shape[1] / self.cr)

        dtype, shape = a.dtype, a.shape

        self.R = self._gen_R(a.shape[1], self.k)
        projected = np.matmul(a, self.R)

        return projected

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
        b = numcodecs.compat.ensure_ndarray(buf)
        
        projected = b.reshape(b.shape[0], self.k)
        decoded = np.matmul(projected, self.R.T).astype(np.float32)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(RPCodec)
