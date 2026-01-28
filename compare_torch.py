import time
import numpy as np

# Test with PyTorch
try:
    import torch

    # Create same data
    np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5

    # PyTorch version
    torch_tensor = torch.from_numpy(np_array.copy())

    # Warmup
    _ = torch_tensor.sum()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(10):
        result = torch_tensor.sum()
    end = time.perf_counter()
    torch_time = (end - start) / 10 * 1000  # ms

    print(f"PyTorch sum: {result.item():.2f}")
    print(f"PyTorch time: {torch_time:.2f} ms")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # Now test with single thread
    torch.set_num_threads(1)
    start = time.perf_counter()
    for _ in range(10):
        result_st = torch_tensor.sum()
    end = time.perf_counter()
    torch_time_st = (end - start) / 10 * 1000  # ms

    print(f"\nPyTorch (1 thread) time: {torch_time_st:.2f} ms")
    print(f"Speedup from multi-threading: {torch_time_st/torch_time:.1f}x")

except ImportError:
    print("PyTorch not installed")

# Test with NumPy
np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5

# Warmup
_ = np_array.sum()

# Benchmark NumPy
start = time.perf_counter()
for _ in range(10):
    result = np_array.sum()
end = time.perf_counter()
numpy_time = (end - start) / 10 * 1000  # ms

print(f"\nNumPy sum: {result:.2f}")
print(f"NumPy time: {numpy_time:.2f} ms")

# Compare to tinygrad
print(f"\nTinygrad time (from your test): ~2.35 ms")
print(f"NumPy vs Tinygrad: {numpy_time/2.35:.1f}x")
