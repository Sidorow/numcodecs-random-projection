from pathlib import Path

import numcodecs
import numcodecs.registry
import numpy as np
import pytest
import xarray as xr

TEST_DIR = Path(__file__).parent
TEST_DATA_PATH = TEST_DIR / "data" / "test_data.nc"

DATA = xr.open_dataset(TEST_DATA_PATH)
TEST_DATA = DATA.t.squeeze().values


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))
    print(codec.__class__.__name__)
    assert codec.__class__.__name__ == "RPCodec"
    assert codec.__class__.__module__ == "numcodecs_random_projection"
    assert codec.cr == 10.0

    codec2 = numcodecs.registry.get_codec(dict(id="rp", k=20))
    print(codec2.__class__.__name__)
    assert codec2.__class__.__name__ == "RPCodec"
    assert codec2.__class__.__module__ == "numcodecs_random_projection"
    assert codec2.k == 20


def check_roundtrip(data: np.ndarray):
    codec_dct = numcodecs.registry.get_codec(dict(id="rp", method="dct", cr=10.0))
    codec_gaussian = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0)
    )

    encoded = codec_dct.encode(data)
    decoded = codec_dct.decode(encoded)
    assert data.shape == decoded.shape
    assert data.dtype == decoded.dtype

    encoded = codec_gaussian.encode(data)
    decoded = codec_gaussian.decode(encoded)
    assert data.shape == decoded.shape
    assert data.dtype == decoded.dtype


def test_roundtrip():
    # Test with a small dataset
    data = np.copy(TEST_DATA)
    check_roundtrip(data)


def test_seed():
    # Test that same seed produces same results
    data = np.copy(TEST_DATA)

    codec1 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42)
    )
    codec2 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42)
    )

    encoded1 = codec1.encode(data.copy())
    encoded2 = codec2.encode(data.copy())

    assert encoded1 == encoded2

    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=10.0, seed=43))

    encoded3 = codec3.encode(data.copy())

    assert encoded1 != encoded3


def test_invalid_codec():
    # Test that missing or invalid parameters raises error
    with pytest.raises(
        ValueError, match="Parameter 'cr' or 'k' must be specified for RPCodec."
    ):
        numcodecs.registry.get_codec(dict(id="rp"))

    with pytest.raises(ValueError, match=r"Unknown method"):
        numcodecs.registry.get_codec(dict(id="rp", method="invalid_method"))


def test_robustness():
    codec2 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5))
    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5, k=20))

    data = np.copy(TEST_DATA)

    # Should correctly calculate k from cr (180 / 9.5 = 18.9 -> 19)
    codec2.encode(data)
    assert codec2.k == 19

    # Should use k over cr when both are specified
    codec3.encode(data)
    assert codec3.k == 20
