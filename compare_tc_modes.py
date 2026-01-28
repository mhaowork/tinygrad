#!/usr/bin/env python3
"""
Compare different tensor core modes to show performance impact
Shows: No TC vs Current WMMA vs Emulated
Measures actual GPU kernel execution time (not Python overhead)
"""
import os
os.environ["CUDA"] = "1"

from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import lower_schedule

def bench_tc_mode(M, N, K, tc_mode, iterations=10):
  """Benchmark with specific TC mode using actual GPU kernel timing"""
  os.environ["TC"] = str(tc_mode)
  
  # Create and realize tensors
  A = Tensor.rand(M, K, dtype=dtypes.float16).realize()
  B = Tensor.rand(K, N, dtype=dtypes.float16).realize()
  
  # Get the kernel execution item
  C = A @ B
  sched = C.schedule()
  lowered = list(lower_schedule(sched))
  ei = lowered[-1][1]  # ExecItem for matmul
  
  # Warmup
  for _ in range(2):
    ei.run(wait=True)
  
  # Benchmark - get actual GPU kernel time
  times = []
  for _ in range(iterations):
    kernel_time = ei.run(wait=True)
    times.append(kernel_time)
  
  min_time = min(times)
  flops = 2 * M * N * K
  tflops = (flops / min_time) / 1e12
  
  return min_time, tflops

def main():
  M = N = K = 4096
  
  print("=" * 80)
  print(f"Comparing Tensor Core Modes: {M}x{N}x{K} FP16 Matmul")
  print("=" * 80)
  
  modes = [
    (0, "TC=0: No Tensor Cores (Pure CUDA Cores)"),
    (1, "TC=1: WMMA Tensor Cores (Current Implementation)"),
    (3, "TC=3: Emulated TC with Local Memory"),
  ]
  
  results = []
  
  for tc_mode, description in modes:
    print(f"\n{description}")
    print("-" * 80)
    
    try:
      min_time, tflops = bench_tc_mode(M, N, K, tc_mode, iterations=15)
      
      print(f"  Time:   {min_time*1e6:8.2f} μs")
      print(f"  TFLOPS: {tflops:8.2f}")
      
      results.append((tc_mode, description, min_time, tflops))
      
    except Exception as e:
      print(f"  ERROR: {e}")
      results.append((tc_mode, description, None, None))
  
  # Summary
  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(f"{'Mode':<8} {'Description':<45} {'Time (μs)':<12} {'TFLOPS':<10} {'Speedup':<10}")
  print("-" * 80)
  
  baseline_time = results[0][2] if results[0][2] else 1.0
  
  for tc_mode, desc, min_time, tflops in results:
    if min_time is not None:
      speedup = baseline_time / min_time
      desc_short = desc.split(":")[1].strip()[:40]
      print(f"TC={tc_mode:<5} {desc_short:<45} {min_time*1e6:>11.2f} {tflops:>9.2f}  {speedup:>9.2f}x")
    else:
      desc_short = desc.split(":")[1].strip()[:40]
      print(f"TC={tc_mode:<5} {desc_short:<45} {'FAILED':<11} {'N/A':<9}  {'N/A':<9}")
  
  print("\n" + "=" * 80)
  
  # Show best
  valid_results = [(tc, d, t, tf) for tc, d, t, tf in results if t is not None]
  if valid_results:
    best = max(valid_results, key=lambda x: x[3])
    print(f"Best Performance: TC={best[0]} with {best[3]:.2f} TFLOPS ({best[2]*1e6:.2f} μs)")
    
    # Show speedup of WMMA over no TC
    if len(valid_results) >= 2 and valid_results[0][2] and valid_results[1][2]:
      speedup = valid_results[0][2] / valid_results[1][2]
      print(f"\nWMMA Speedup over CUDA Cores: {speedup:.2f}x")
      print(f"  Without TC: {valid_results[0][3]:.2f} TFLOPS")
      print(f"  With TC:    {valid_results[1][3]:.2f} TFLOPS")
  
  print("\n" + "=" * 80)
  print("Note: The current implementation (TC=1) uses basic WMMA without advanced")
  print("optimizations. PR #13511 adds 3-stage pipelining and swizzled memory to")
  print("achieve ~309 TFLOPS (1.5-2x improvement over current WMMA).")
  print("=" * 80)

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n\nBenchmark interrupted by user")
  except Exception as e:
    print(f"\n\nError: {e}")
    import traceback
    traceback.print_exc()

