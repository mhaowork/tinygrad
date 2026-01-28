#!/usr/bin/env python3
"""
Benchmark tinygrad matmul TFLOPS on CUDA (current implementation)
Measures performance with different matrix sizes and configurations
"""
import os, time
import numpy as np

# Force CUDA backend
os.environ["CUDA"] = "1"

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv, colored

def benchmark_matmul(M, N, K, dtype_in=dtypes.float16, dtype_out=dtypes.float, 
                     warmup=3, iterations=20, use_tc=1):
  """
  Benchmark matrix multiplication: C[M,N] = A[M,K] @ B[K,N]
  
  Args:
    M, N, K: Matrix dimensions
    dtype_in: Input dtype (half or float)
    dtype_out: Output dtype (half or float)
    warmup: Number of warmup iterations
    iterations: Number of timed iterations
    use_tc: Tensor core setting (0=disabled, 1=enabled, 2=shape only, 3=emulated)
  """
  print(f"\n{'='*70}")
  print(f"Benchmarking: M={M}, N={N}, K={K}")
  print(f"Input dtype: {dtype_in}, Output dtype: {dtype_out}")
  print(f"Tensor Cores: {'ENABLED' if use_tc == 1 else 'DISABLED' if use_tc == 0 else f'TC={use_tc}'}")
  print(f"{'='*70}")
  
  # Set tensor core mode
  old_tc = os.environ.get("TC", "1")
  os.environ["TC"] = str(use_tc)
  
  # Create tensors on CUDA and realize them
  A = Tensor.rand(M, K, dtype=dtype_in, device=f"{Device.DEFAULT}:0").realize()
  B = Tensor.rand(K, N, dtype=dtype_in, device=f"{Device.DEFAULT}:0").realize()
  
  # Get the kernel execution item for accurate timing
  from tinygrad.engine.realize import lower_schedule
  C = A @ B
  sched = C.schedule()
  lowered = list(lower_schedule(sched))
  ei = lowered[-1][1]  # ExecItem for the matmul kernel
  
  # Warmup runs
  print(f"Warming up ({warmup} iterations)...")
  for _ in range(warmup):
    ei.run(wait=True)
  
  # Timed runs - measure actual GPU kernel time
  print(f"Running benchmark ({iterations} iterations)...")
  times = []
  
  for i in range(iterations):
    kernel_time = ei.run(wait=True)  # Returns GPU kernel execution time
    times.append(kernel_time)
    
    # Show progress
    if (i + 1) % 5 == 0:
      print(f"  Completed {i+1}/{iterations} iterations...")
  
  # Restore TC setting
  os.environ["TC"] = old_tc
  
  # Calculate statistics
  times = np.array(times)
  min_time = np.min(times)
  mean_time = np.mean(times)
  median_time = np.median(times)
  std_time = np.std(times)
  
  # Calculate FLOPS (matrix multiply: 2*M*N*K operations)
  flops = 2 * M * N * K
  tflops_min = (flops / min_time) / 1e12
  tflops_mean = (flops / mean_time) / 1e12
  tflops_median = (flops / median_time) / 1e12
  
  # Calculate bandwidth (bytes transferred)
  # A: M*K elements, B: K*N elements, C: M*N elements
  bytes_transferred = (M*K + K*N) * dtype_in.itemsize + M*N * dtype_out.itemsize
  bandwidth_gb_s = (bytes_transferred / min_time) / 1e9
  
  # Print results
  print(f"\n{colored('Results:', 'green')}")
  print(f"  Min time:    {min_time*1e6:8.2f} μs")
  print(f"  Mean time:   {mean_time*1e6:8.2f} μs")
  print(f"  Median time: {median_time*1e6:8.2f} μs")
  print(f"  Std dev:     {std_time*1e6:8.2f} μs")
  print(f"\n{colored('Performance:', 'cyan')}")
  print(f"  Peak TFLOPS:   {colored(f'{tflops_min:7.2f}', 'yellow')} TFLOPS")
  print(f"  Mean TFLOPS:   {tflops_mean:7.2f} TFLOPS")
  print(f"  Median TFLOPS: {tflops_median:7.2f} TFLOPS")
  print(f"  Bandwidth:     {bandwidth_gb_s:7.2f} GB/s")
  print(f"  Arithmetic Intensity: {flops/bytes_transferred:7.2f} FLOPS/byte")
  
  return {
    'M': M, 'N': N, 'K': K,
    'dtype_in': str(dtype_in),
    'dtype_out': str(dtype_out),
    'use_tc': use_tc,
    'min_time_us': min_time * 1e6,
    'mean_time_us': mean_time * 1e6,
    'tflops_peak': tflops_min,
    'tflops_mean': tflops_mean,
    'bandwidth_gb_s': bandwidth_gb_s
  }

if __name__ == "__main__":
  print(f"\n{colored('Tinygrad CUDA Matmul TFLOPS Benchmark', 'BLUE')}")
  print(f"Device: {Device.DEFAULT}")
  
  # Check CUDA is available
  if "CUDA" not in Device.DEFAULT:
    print(colored("ERROR: CUDA not available. Please set CUDA=1", "red"))
    exit(1)
  
  # Get GPU info
  from tinygrad.runtime.ops_cuda import CUDADevice
  try:
    device = CUDADevice(f"{Device.DEFAULT}:0")
    print(f"GPU Architecture: {device.arch}")
  except:
    print("Could not get GPU info")
  
  results = []
  
  # Configuration
  warmup = getenv("WARMUP", 3)
  iterations = getenv("ITERS", 20)
  use_tc = getenv("TC", 1)
  
  # Test different sizes
  test_configs = [
    # Small matrices
    (512, 512, 512, "Small (512³)"),
    (1024, 1024, 1024, "Medium (1024³)"),
    (2048, 2048, 2048, "Large (2048³)"),
    (4096, 4096, 4096, "XLarge (4096³)"),
  ]
  
  # Test square matrices with FP16 input, FP32 accumulation (typical for ML)
  print(f"\n{colored('=== FP16 Input, FP32 Output (Typical ML Workload) ===', 'MAGENTA')}")
  for M, N, K, desc in test_configs:
    result = benchmark_matmul(
      M, N, K, 
      dtype_in=dtypes.float16,
      dtype_out=dtypes.float,
      warmup=warmup,
      iterations=iterations,
      use_tc=use_tc
    )
    results.append(result)
  
  # Test FP16 input and output (maximum tensor core speed)
  print(f"\n{colored('=== FP16 Input, FP16 Output (Max Tensor Core Speed) ===', 'MAGENTA')}")
  for M, N, K, desc in test_configs:
    result = benchmark_matmul(
      M, N, K,
      dtype_in=dtypes.float16,
      dtype_out=dtypes.float16,
      warmup=warmup,
      iterations=iterations,
      use_tc=use_tc
    )
    results.append(result)
  
  # Summary table
  print(f"\n{colored('='*70, 'BLUE')}")
  print(f"{colored('SUMMARY TABLE', 'BLUE')}")
  print(f"{colored('='*70, 'BLUE')}")
  print(f"{'Size':<12} {'In→Out':<12} {'TC':<4} {'Peak TFLOPS':<12} {'Mean TFLOPS':<12} {'BW (GB/s)':<12}")
  print(f"{'-'*70}")
  
  for r in results:
    size_str = f"{r['M']}x{r['N']}x{r['K']}"
    dtype_str = f"{r['dtype_in'].split('.')[-1]}→{r['dtype_out'].split('.')[-1]}"
    tc_str = "ON" if r['use_tc'] == 1 else "OFF"
    print(f"{size_str:<12} {dtype_str:<12} {tc_str:<4} "
          f"{r['tflops_peak']:>11.2f} {r['tflops_mean']:>11.2f} {r['bandwidth_gb_s']:>11.2f}")
  
  # Find best performance
  best = max(results, key=lambda x: x['tflops_peak'])
  print(f"\n{colored('Best Performance:', 'GREEN')}")
  tflops_str = f"{best['tflops_peak']:.2f} TFLOPS"
  print(f"  {best['M']}x{best['N']}x{best['K']} "
        f"({best['dtype_in'].split('.')[-1]}→{best['dtype_out'].split('.')[-1]}): "
        f"{colored(tflops_str, 'YELLOW')}")
  
  print(f"\n{colored('Benchmark Complete!', 'GREEN')}")
  print(f"\nTo modify benchmark:")
  print(f"  WARMUP=5 ITERS=50 TC=1 python3 benchmark_matmul_tflops.py")
  print(f"  TC=0 - Disable tensor cores")
  print(f"  TC=1 - Enable tensor cores (default)")
  print(f"  TC=2 - Tensor core shapes without WMMA")
  print(f"  TC=3 - Emulate tensor cores with locals")

