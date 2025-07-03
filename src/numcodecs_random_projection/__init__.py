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
from typing_extensions import Buffer  # MSPV 3.12


class RandomProjectionCodec(Codec):
    """
    Codec that decodes to an all-zero array of the same data type and shape as
    the original data.

    Encoding produces a bytestring containing information on the data type and
    shape.
    """

    __slots__ = ()

    codec_id: str = "rp"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded `buf`fer information as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        # message: dtype shape
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Encoded buffer information. Must be an object representing a
            bytestring, e.g. [`bytes`][bytes] or a 1D array of
            [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The decoded data will be all-zero after decoding.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))
        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )

        decoded = np.zeros(shape, dtype)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(RandomProjectionCodec)
