#!/usr/bin/env python3
"""
Simple quick matmul benchmark for tinygrad on CUDA
Measures actual GPU kernel execution time (not Python overhead)
Usage: CUDA=1 python3 benchmark_matmul_simple.py
"""
import os
os.environ["CUDA"] = "1"

from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import ExecItem, lower_schedule

def quick_bench(M=4096, N=4096, K=4096, dtype=dtypes.float16, iterations=10):
  print(f"Benchmarking {M}x{N}x{K} matmul with {dtype}")
  print(f"Device: {Device.DEFAULT}")
  
  # Create tensors and realize them
  A = Tensor.rand(M, K, dtype=dtype).realize()
  B = Tensor.rand(K, N, dtype=dtype).realize()
  
  # Get the kernel execution item
  C = A @ B
  sched = C.schedule()
  si = sched[-1]  # Last schedule item is the matmul kernel
  
  # Create ExecItem for direct kernel timing
  lowered = list(lower_schedule(sched))
  ei = lowered[-1][1]  # Get the ExecItem for matmul
  
  # Warmup
  print("Warming up...")
  for _ in range(3):
    ei.run(wait=True)
  
  # Benchmark - get actual GPU kernel time
  print(f"Running {iterations} iterations...")
  times = []
  for i in range(iterations):
    kernel_time = ei.run(wait=True)  # Returns actual GPU time in seconds
    times.append(kernel_time)
    print(f"  Iteration {i+1}/{iterations}: {kernel_time*1e6:.2f} μs")
  
  # Results
  min_time = min(times)
  avg_time = sum(times) / len(times)
  
  flops = 2 * M * N * K
  tflops_peak = (flops / min_time) / 1e12
  tflops_avg = (flops / avg_time) / 1e12
  
  bytes_xfer = (M*K + K*N + M*N) * dtype.itemsize
  bandwidth = (bytes_xfer / min_time) / 1e9
  
  print(f"\nResults:")
  print(f"  Min time:  {min_time*1e6:8.2f} μs")
  print(f"  Avg time:  {avg_time*1e6:8.2f} μs")
  print(f"  Peak:      {tflops_peak:8.2f} TFLOPS")
  print(f"  Average:   {tflops_avg:8.2f} TFLOPS")
  print(f"  Bandwidth: {bandwidth:8.2f} GB/s")
  
  return tflops_peak

if __name__ == "__main__":
  import sys
  
  # Parse args: M N K [dtype]
  M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
  N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
  K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
  
  dtype_map = {"half": dtypes.float16, "float": dtypes.float32, "float16": dtypes.float16, "float32": dtypes.float32}
  dtype = dtype_map.get(sys.argv[4], dtypes.float16) if len(sys.argv) > 4 else dtypes.float16
  
  quick_bench(M, N, K, dtype)
  
  print("\nUsage: python3 benchmark_matmul_simple.py [M] [N] [K] [dtype]")
  print("Example: python3 benchmark_matmul_simple.py 2048 2048 2048 half")

