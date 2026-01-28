#!/usr/bin/env python3
"""Test actual memory bandwidth achieved"""

import time
import numpy as np
from tinygrad import Tensor

SIZE = 2048
ELEMENTS = SIZE * SIZE
BYTES = ELEMENTS * 4

print(f"Testing {SIZE}x{SIZE} sum ({BYTES/1024/1024:.1f} MB)")

a_np = np.random.random((SIZE, SIZE)).astype(np.float32)
a = Tensor(a_np).realize()

# Warmup
for _ in range(10):
    result = a.sum().realize()

# Benchmark
times = []
for _ in range(50):
    start = time.perf_counter()
    result = a.sum().realize()
    times.append(time.perf_counter() - start)

median_time = np.median(times)
avg_time = np.mean(times)
min_time = np.min(times)

print(f"\nTiming:")
print(f"  Median: {median_time*1000:.2f} ms")
print(f"  Average: {avg_time*1000:.2f} ms")
print(f"  Best: {min_time*1000:.2f} ms")

bandwidth_median = BYTES / median_time / 1e9
bandwidth_best = BYTES / min_time / 1e9

print(f"\nBandwidth:")
print(f"  Median: {bandwidth_median:.1f} GB/s")
print(f"  Best: {bandwidth_best:.1f} GB/s")
print(f"  Peak M1/M2: ~200 GB/s")
print(f"  Utilization: {bandwidth_median/200*100:.1f}%")
