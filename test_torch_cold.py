import torch
import time
import numpy as np

# Force single thread
torch.set_num_threads(1)

print("Testing PyTorch cold vs warm performance...\n")

# Create tensor
x = torch.randn(4096, 4096, dtype=torch.float32)

# Test cold start (first run)
start = time.perf_counter()
result = x.sum()
cold_time = (time.perf_counter() - start) * 1000

bandwidth_cold = (4096 * 4096 * 4 / (cold_time / 1000)) / 1e9

print(f"Cold (1st run):  {cold_time:.2f} ms, {bandwidth_cold:.1f} GB/s")

# Test warm runs
times = []
for i in range(9):
    start = time.perf_counter()
    result = x.sum()
    times.append((time.perf_counter() - start) * 1000)

warm_time = sum(times) / len(times)
bandwidth_warm = (4096 * 4096 * 4 / (warm_time / 1000)) / 1e9

print(f"Warm (2-10):     {warm_time:.2f} ms, {bandwidth_warm:.1f} GB/s")
print(f"\nSpeedup: {cold_time / warm_time:.2f}x")
