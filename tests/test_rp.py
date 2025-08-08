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


def test_roundtrip_blocks():
    # Test with a small dataset using blocks
    data = np.copy(TEST_DATA)
    codec_dct = numcodecs.registry.get_codec(dict(id="rp", method="dct", k=20))
    codec_gaussian = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", k=20)
    )

    # Force block usage by calling the block methods directly
    projected_dct = codec_dct._project_blocks(
        data, data.shape[1], 20, data.dtype, block_size=10
    )

    projected_gaussian = codec_gaussian._project_blocks(
        data, data.shape[1], 20, data.dtype, block_size=10
    )

    reconstructed_dct = codec_dct._reconstruct_blocks(
        projected_dct, data.shape[1], 20, data.dtype, block_size=10, seed=codec_dct.seed
    )

    reconstructed_gaussian = codec_gaussian._reconstruct_blocks(
        projected_gaussian,
        data.shape[1],
        20,
        data.dtype,
        block_size=10,
        seed=codec_gaussian.seed,
    )

    assert reconstructed_dct.shape == data.shape
    assert reconstructed_dct.dtype == data.dtype

    assert reconstructed_gaussian.shape == data.shape
    assert reconstructed_gaussian.dtype == data.dtype


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


def test_seed_blocks():
    # Test that block methods produce same results for Gaussian method with same seed
    # Different seeds should produce different results
    codec1 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42)
    )
    codec2 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=42)
    )
    codec3 = numcodecs.registry.get_codec(
        dict(id="rp", method="gaussian", cr=10.0, seed=43)
    )

    data = np.random.randn(100, 50).astype(np.float64)

    projected_blocks1 = codec1._project_blocks(
        np.copy(data), 50, 10, data.dtype, block_size=5
    )
    projected_blocks2 = codec2._project_blocks(
        np.copy(data), 50, 10, data.dtype, block_size=5
    )
    projected_blocks3 = codec3._project_blocks(
        np.copy(data), 50, 10, data.dtype, block_size=5
    )

    assert np.array_equal(projected_blocks1, projected_blocks2)
    assert not np.array_equal(projected_blocks1, projected_blocks3)

    reconstructed_blocks1 = codec1._reconstruct_blocks(
        projected_blocks1, 50, 10, data.dtype, block_size=5, seed=codec1.seed
    )

    reconstructed_blocks2 = codec2._reconstruct_blocks(
        projected_blocks2, 50, 10, data.dtype, block_size=5, seed=codec2.seed
    )

    reconstructed_blocks3 = codec3._reconstruct_blocks(
        projected_blocks3, 50, 10, data.dtype, block_size=5, seed=codec3.seed
    )

    assert np.array_equal(reconstructed_blocks1, reconstructed_blocks2)
    assert not np.array_equal(reconstructed_blocks1, reconstructed_blocks3)


def test_reconstruct_seed():
    # Test that reconstruction can be done with a different codec
    # Should produce same reconstruction as original codec from which data was encoded
    # Use large data to trigger block processing
    small_data = np.random.randn(100, 50).astype(np.float64)
    large_data = np.random.randn(100, 3000).astype(np.float64)

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


def test_block_vs_full_matrix_dct():
    # Test that block and full matrix methods produce same results for DCT method
    # Both methods should produce (numerically) identical results for the same data in float64
    codec = numcodecs.registry.get_codec(dict(id="rp", method="dct", cr=5.0))

    data = np.random.randn(100, 50).astype(np.float64)

    projected_blocks = codec._project_blocks(data, 50, 10, data.dtype, block_size=5)

    full_R = codec._gen_R(50, 10, data.dtype)
    projected_full = np.matmul(data, full_R)

    np.testing.assert_allclose(projected_blocks, projected_full, atol=1e-15)
    assert projected_blocks.shape == (100, 10) and projected_full.shape == (100, 10)

    reconstructed_blocks = codec._reconstruct_blocks(
        projected_blocks, 50, 10, data.dtype, block_size=5, seed=codec.seed
    )

    reconstructed_full = np.matmul(projected_blocks, full_R.T)

    np.testing.assert_allclose(reconstructed_blocks, reconstructed_full, atol=1e-15)
    assert reconstructed_blocks.shape == (100, 50) and reconstructed_full.shape == (
        100,
        50,
    )


def test_invalid_codec():
    # Test that missing or invalid parameters raises error
    with pytest.raises(
        ValueError, match="Parameter 'cr' or 'k' must be specified for RPCodec."
    ):
        numcodecs.registry.get_codec(dict(id="rp"))

    with pytest.raises(ValueError, match=r"Unknown method"):
        numcodecs.registry.get_codec(dict(id="rp", method="invalid_method"))


def test_invalid_data():
    # Test that non-floating-point data raises error
    codec = numcodecs.registry.get_codec(dict(id="rp", cr=10.0))

    # Test with integer data
    with pytest.raises(ValueError, match=r"RPCodec requires .* floating-point data"):
        codec.encode(np.random.randint(50, 100, size=(3, 4)))

    # Test with non-2D data
    with pytest.raises(ValueError, match=r"RPCodec requires 2D .* data"):
        codec.encode(np.random.randn(50, 100, 3))


def test_robustness():
    codec2 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5))
    codec3 = numcodecs.registry.get_codec(dict(id="rp", cr=9.5, k=20))
    codec4 = numcodecs.registry.get_codec(dict(id="rp", cr=10))
    codec5 = numcodecs.registry.get_codec(dict(id="rp", cr=10))

    data = np.copy(TEST_DATA)

    # Create NaN and Inf data
    nan_data = np.copy(data)
    nan_data.fill(np.nan)

    inf_data = np.copy(data)
    inf_data.fill(np.inf)

    # Should correctly calculate k from cr (180 / 9.5 = 18.9 -> 19)
    codec2.encode(data)
    assert codec2.k == 19

    # Should use k over cr when both are specified
    codec3.encode(data)
    assert codec3.k == 20

    # Should handle NaN and Inf values correctly
    # NaN and inf values should be replaced with 0.0
    nan_encode = codec4.encode(nan_data)
    nan_decoded = codec4.decode(nan_encode)
    assert not np.isnan(nan_decoded).any()
    assert not np.any(nan_decoded)

    inf_encode = codec5.encode(inf_data)
    inf_decoded = codec5.decode(inf_encode)
    assert not np.isinf(inf_decoded).any()
    assert not np.any(inf_decoded)
