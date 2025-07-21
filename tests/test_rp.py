import numcodecs
import numcodecs.registry
import numpy as np
import pytest


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
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert data.shape == decoded.shape
    assert data.dtype == decoded.dtype


def test_roundtrip():
    # Test with a simple 2D array
    data = np.random.randn(1000, 500)
    check_roundtrip(data)

    # Test with a high-dimensional array
    data = np.random.randn(500, 10000)
    check_roundtrip(data)


def test_seed():
    # Test that same seed produces same results
    data = np.random.rand(50, 100)

    codec1 = numcodecs.registry.get_codec(dict(id="rp", cr=10.0, seed=42))
    codec2 = numcodecs.registry.get_codec(dict(id="rp", cr=10.0, seed=42))

    encoded1 = codec1.encode(data.copy())
    encoded2 = codec2.encode(data.copy())

    assert encoded1 == encoded2

    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=10.0, seed=43))

    encoded3 = codec3.encode(data.copy())

    assert encoded1 != encoded3


def test_invalid_codec():
    # Test that missing parameters raises error
    with pytest.raises(
        ValueError, match="Parameter 'cr' or 'k' must be specified for RPCodec."
    ):
        numcodecs.registry.get_codec(dict(id="rp"))


def test_robustness():
    codec2 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5))
    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5, k=20))

    data = np.random.rand(50, 100)

    # Should correctly calculate k from cr
    codec2.encode(data)
    assert codec2.k == 11

    # Should use k over cr when both are specified
    codec3.encode(data)
    assert codec3.k == 20
