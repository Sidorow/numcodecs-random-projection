#!/usr/bin/env python3
"""
Benchmark script to test sparse vs dense random projection performance.
"""

import time

import numpy as np

from numcodecs_random_projection import RPCodec


def benchmark_methods(n_samples=1000, n_features=5000, k=100, n_runs=5):
    """Compare performance of sparse vs dense random projection methods."""

    print(f"Benchmarking with data shape: ({n_samples}, {n_features})")
    print(f"Projection dimension k: {k}")
    print(f"Number of runs: {n_runs}")
    print("-" * 50)

    # Generate test data
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    # Test Gaussian method
    codec_gaussian = RPCodec(k=k, method="gaussian", seed=42)

    gaussian_times = []
    for i in range(n_runs):
        start_time = time.time()
        encoded = codec_gaussian.encode(data)
        end_time = time.time()
        gaussian_times.append(end_time - start_time)

    avg_gaussian = np.mean(gaussian_times)
    std_gaussian = np.std(gaussian_times)

    # Test Sparse method
    codec_sparse = RPCodec(k=k, method="dct", seed=42)

    sparse_times = []
    for i in range(n_runs):
        start_time = time.time()
        encoded = codec_sparse.encode(data)
        end_time = time.time()
        sparse_times.append(end_time - start_time)

    avg_sparse = np.mean(sparse_times)
    std_sparse = np.std(sparse_times)

    # Print results
    print(f"Gaussian method: {avg_gaussian:.4f} ± {std_gaussian:.4f} seconds")
    print(f"DCT method:   {avg_sparse:.4f} ± {std_sparse:.4f} seconds")
    print(f"Speedup factor:  {avg_gaussian / avg_sparse:.2f}x")

    if avg_sparse < avg_gaussian:
        improvement = ((avg_gaussian - avg_sparse) / avg_gaussian) * 100
        print(f"DCT method is {improvement:.1f}% faster")
    else:
        slowdown = ((avg_sparse - avg_gaussian) / avg_gaussian) * 100
        print(f"DCT method is {slowdown:.1f}% slower")


if __name__ == "__main__":
    print("Random Projection Performance Benchmark")
    print("=" * 50)

    # Test different scenarios
    benchmark_methods(n_samples=1000, n_features=5000, k=100)
    print()
    benchmark_methods(n_samples=500, n_features=10000, k=50)
