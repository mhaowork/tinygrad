#!/usr/bin/env python3
"""Test if PyTorch is using multi-threading for sum"""

import torch
import numpy as np
import time

def benchmark(a_torch, num_threads):
    torch.set_num_threads(num_threads)

    # Warmup
    for _ in range(10):
        result = a_torch.sum().item()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        result = a_torch.sum().item()
    end = time.perf_counter()

    avg_time = (end - start) / 100 * 1000  # ms
    return avg_time, result

if __name__ == "__main__":
    for size in [2048, 4096]:
        print(f"\n{'='*60}")
        print(f"Testing {size}x{size}")
        print(f"{'='*60}")

        a = np.random.random((size, size)).astype(np.float32)
        a_torch = torch.from_numpy(a)

        for num_threads in [1, 2, 4, 8]:
            time_ms, result = benchmark(a_torch, num_threads)
            print(f"{num_threads} thread(s): {time_ms:6.2f} ms")
