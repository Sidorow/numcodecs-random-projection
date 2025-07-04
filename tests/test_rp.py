import numcodecs
import numcodecs.registry
import numpy as np


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


def test_roundtrip():
    # Test with a simple 2D array
    data = np.random.randn(100, 50)
    check_roundtrip(data)

    # Test with a high-dimensional array
    data = np.random.randn(50, 1000)
    check_roundtrip(data)

    # Test with a small array
    data = np.random.randn(5, 10)
    check_roundtrip(data)

    # Test with a single row
    data = np.random.randn(1, 50)
    check_roundtrip(data)

    # Test with integer data
    data = np.random.randint(0, 100, size=(20, 30))
    check_roundtrip(data)
