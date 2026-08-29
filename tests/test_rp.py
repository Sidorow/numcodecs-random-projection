from pathlib import Path

import numcodecs
import numcodecs.registry
import numpy as np
import pytest
import xarray as xr

from numcodecs_random_projection._mt_rng import MultithreadedRNG

TEST_DIR = Path(__file__).parent
TEST_DATA_PATH = TEST_DIR / "data" / "test_data.nc"

DATA = xr.open_dataset(TEST_DATA_PATH)
TEST_DATA = DATA.t.squeeze().values


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))
    assert codec.__class__.__name__ == "RPCodec"
    assert codec.__class__.__module__ == "numcodecs_random_projection"
    assert codec._cr == 10.0
    assert repr(codec) == "RPCodec(cr=10.0, method='dct', debug=False)"

    codec2 = numcodecs.registry.get_codec(dict(id="rp", k=20))
    assert codec2.__class__.__name__ == "RPCodec"
    assert codec2.__class__.__module__ == "numcodecs_random_projection"
    assert codec2._k == 20
    assert repr(codec2) == "RPCodec(k=20, method='dct', debug=False)"


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
    check_roundtrip(TEST_DATA)


def test_roundtrip_blocks():
    # Test with a small dataset using blocks
    data = TEST_DATA
    codec_dct = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="dct",
            k=20,
            # force block_size=10
            max_block_memory=data.shape[1] * data.itemsize * 10,
        )
    )
    codec_gaussian = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="gaussian",
            k=20,
            # force block_size=10
            max_block_memory=data.shape[1] * data.itemsize * 10,
        )
    )

    projected_dct = codec_dct.encode(data)
    reconstructed_dct = codec_dct.decode(projected_dct)

    projected_gaussian = codec_gaussian.encode(data)
    reconstructed_gaussian = codec_gaussian.decode(projected_gaussian)

    assert reconstructed_dct.shape == data.shape
    assert reconstructed_dct.dtype == data.dtype

    assert reconstructed_gaussian.shape == data.shape
    assert reconstructed_gaussian.dtype == data.dtype


def test_seed():
    # Test that same seed and block memory produces same results
    data = TEST_DATA

    codec1 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42, max_block_memory=2**26)
    )
    codec2 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42, max_block_memory=2**26)
    )

    encoded1 = codec1.encode(data)
    encoded2 = codec2.encode(data)

    assert encoded1 == encoded2

    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=10.0, seed=43))

    encoded3 = codec3.encode(data)

    assert encoded1 != encoded3


def test_seed_blocks():
    # Test that block methods produce same results for Gaussian method with same seed
    # Different seeds should produce different results
    rng = np.random.default_rng()
    data = rng.standard_normal(size=(100, 50), dtype=np.float64)

    codec1 = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="gaussian",
            cr=10.0,
            seed=42,
            # force block_size=5
            max_block_memory=data.shape[1] * data.itemsize * 5,
        )
    )
    codec2 = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="gaussian",
            cr=10.0,
            seed=42,
            # force block_size=5
            max_block_memory=data.shape[1] * data.itemsize * 5,
        )
    )
    codec3 = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="gaussian",
            cr=10.0,
            seed=43,
            # force block_size=5
            max_block_memory=data.shape[1] * data.itemsize * 5,
        )
    )

    projected_blocks1 = codec1.encode(data)
    projected_blocks2 = codec2.encode(data)
    projected_blocks3 = codec3.encode(data)

    assert np.array_equal(projected_blocks1, projected_blocks2)
    assert not np.array_equal(projected_blocks1, projected_blocks3)

    reconstructed_blocks1 = codec1.decode(projected_blocks1)
    reconstructed_blocks2 = codec1.decode(projected_blocks2)
    reconstructed_blocks3 = codec1.decode(projected_blocks3)

    assert np.array_equal(reconstructed_blocks1, reconstructed_blocks2)
    assert not np.array_equal(reconstructed_blocks1, reconstructed_blocks3)


def test_reconstruct_seed():
    # Test that reconstruction can be done with a different codec
    # Should produce same reconstruction as original codec from which data was encoded
    # Use large data to trigger block processing
    rng = np.random.default_rng()
    small_data = rng.standard_normal(size=(100, 50), dtype=np.float64)
    large_data = rng.standard_normal(size=(100, 3000), dtype=np.float64)

    codec1 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", k=1500, seed=42)
    )
    codec2 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=15.0, seed=24)
    )

    encoded_small = codec1.encode(small_data)

    decoded1 = codec1.decode(encoded_small)
    decoded2 = codec2.decode(encoded_small)

    assert np.array_equal(decoded1, decoded2)

    encoded_large = codec1.encode(large_data)

    decoded3 = codec1.decode(encoded_large)
    decoded4 = codec2.decode(encoded_large)

    assert np.array_equal(decoded3, decoded4)


def test_multithreaded_rng_thread_count_invariance():
    # Test that MultithreadedRNG produces identical output regardless of thread count.
    seed = 42
    shape = (2000, 100)

    rng_1t = MultithreadedRNG(seed=seed, threads=1)
    result_1t = np.empty(shape)
    rng_1t.fill_arr(out=result_1t)

    rng_2t = MultithreadedRNG(seed=seed, threads=2)
    result_2t = np.empty(shape)
    rng_2t.fill_arr(out=result_2t)

    rng_4t = MultithreadedRNG(seed=seed, threads=4)
    result_4t = np.empty(shape)
    rng_4t.fill_arr(out=result_4t)

    rng_8t = MultithreadedRNG(seed=seed, threads=8)
    result_8t = np.empty(shape)
    rng_8t.fill_arr(out=result_8t)

    np.testing.assert_array_equal(
        result_1t, result_2t, err_msg="1-thread vs 2-thread RNG output mismatch"
    )
    np.testing.assert_array_equal(
        result_1t, result_4t, err_msg="1-thread vs 4-thread RNG output mismatch"
    )
    np.testing.assert_array_equal(
        result_1t, result_8t, err_msg="1-thread vs 8-thread RNG output mismatch"
    )


def test_block_vs_full_matrix_dct():
    # Test that block and full matrix methods produce same results for DCT method
    # Both methods should produce (numerically) identical results for the same data in float64
    rng = np.random.default_rng()
    data = rng.standard_normal(size=(100, 50), dtype=np.float64)

    full_codec = numcodecs.registry.get_codec(
        dict(id="rp", method="dct", cr=5.0, max_block_memory=-1)
    )
    block_codec = numcodecs.registry.get_codec(
        dict(
            id="rp",
            method="dct",
            cr=5.0,
            # force block_size=5
            max_block_memory=data.shape[1] * data.itemsize * 5,
        )
    )

    projected_full = full_codec.encode(data)
    projected_blocks = block_codec.encode(data)

    reconstructed_full = full_codec.decode(projected_full)
    reconstructed_blocks = block_codec.decode(projected_blocks)

    np.testing.assert_allclose(reconstructed_blocks, reconstructed_full, atol=1e-15)
    assert reconstructed_blocks.shape == (100, 50) and reconstructed_full.shape == (
        100,
        50,
    )


def test_invalid_codec():
    # Test that missing or invalid parameters raises error
    with pytest.raises(
        ValueError, match="exactly one of `mae`, `cr` or `k` must be set"
    ):
        numcodecs.registry.get_codec(dict(id="rp"))

    with pytest.raises(
        ValueError, match="exactly one of `mae`, `cr` or `k` must be set"
    ):
        numcodecs.registry.get_codec(dict(id="rp", mae=0.1, cr=10))

    with pytest.raises(
        ValueError, match="exactly one of `mae`, `cr` or `k` must be set"
    ):
        numcodecs.registry.get_codec(dict(id="rp", mae=0.1, k=1))

    with pytest.raises(
        ValueError, match="exactly one of `mae`, `cr` or `k` must be set"
    ):
        numcodecs.registry.get_codec(dict(id="rp", cr=10, k=1))

    with pytest.raises(ValueError, match=r"unknown method"):
        numcodecs.registry.get_codec(dict(id="rp", k=1, method="invalid_method"))


def test_invalid_data():
    # Test that non-floating-point data raises error
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))

    rng = np.random.default_rng()
    data = rng.integers(50, 100, size=(3, 4))

    # Test with integer data
    with pytest.raises(ValueError, match=r"RPCodec requires floating-point data"):
        codec.encode(data)


def test_nd_data():
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))

    rng = np.random.default_rng()

    for shape in [(), (100,), (50, 100), (50, 100, 3), (25, 50, 3, 4)]:
        encoded = codec.encode(np.asarray(rng.standard_normal(shape)))
        decoded = codec.decode(encoded)
        assert decoded.dtype == np.dtype(float)
        assert decoded.shape == shape


def test_robustness():
    codec1 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5))
    codec2 = numcodecs.registry.get_codec(dict(id="rp", cr=10))

    data = TEST_DATA

    # Create NaN and Inf data
    nan_data = np.copy(data)
    nan_data.fill(np.nan)

    inf_data = np.copy(data)
    inf_data.fill(np.inf)

    # Should correctly calculate k from cr (180 / 9.5 = 18.9 -> 19)
    codec1.encode(data)

    # Should handle NaN and Inf values correctly
    # NaN and inf values should be replaced with 0.0
    nan_encode = codec2.encode(nan_data)
    nan_decoded = codec2.decode(nan_encode)
    assert not np.isnan(nan_decoded).any()
    assert not np.any(nan_decoded)
