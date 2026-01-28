# Tinygrad Matmul TFLOPS Benchmarks

**Accurate GPU kernel timing** - These benchmarks measure actual GPU kernel execution time using `ExecItem.run(wait=True)`, not Python overhead. This matches the methodology used in tinygrad's `extra/gemm/` benchmarks.

## Quick Start

### Simple Benchmark (Fast)
```bash
# Default: 4096x4096x4096 FP16 matmul
CUDA=1 python3 benchmark_matmul_simple.py

# Custom size
CUDA=1 python3 benchmark_matmul_simple.py 2048 2048 2048 half

# FP32 matmul
CUDA=1 python3 benchmark_matmul_simple.py 4096 4096 4096 float32
```

### Comprehensive Benchmark (Detailed)
```bash
# Default: Multiple sizes with FP16 and FP32
CUDA=1 python3 benchmark_matmul_tflops.py

# More iterations for accuracy
CUDA=1 WARMUP=5 ITERS=50 python3 benchmark_matmul_tflops.py

# Test without tensor cores
CUDA=1 TC=0 python3 benchmark_matmul_tflops.py

# Compare tensor core modes
CUDA=1 TC=1 python3 benchmark_matmul_tflops.py  # Normal (WMMA)
CUDA=1 TC=2 python3 benchmark_matmul_tflops.py  # Shape only (no WMMA)
CUDA=1 TC=3 python3 benchmark_matmul_tflops.py  # Emulated with locals
```

## Environment Variables

### benchmark_matmul_simple.py
- `CUDA=1` - Enable CUDA backend (required)

### benchmark_matmul_tflops.py
- `CUDA=1` - Enable CUDA backend (required)
- `TC=0|1|2|3` - Tensor core mode (default: 1)
  - `0` = Disabled
  - `1` = Enabled with WMMA instructions (current way)
  - `2` = Tensor core shapes without WMMA UOps
  - `3` = Emulated with local memory
- `WARMUP=N` - Number of warmup iterations (default: 3)
- `ITERS=N` - Number of timed iterations (default: 20)

## Expected Performance

### NVIDIA RTX 30/40 Series (Ampere/Ada)
- **FP16→FP16**: 150-250 TFLOPS (current implementation)
- **FP16→FP32**: 80-180 TFLOPS (current implementation)
- **FP32→FP32**: 15-40 TFLOPS

### NVIDIA A100
- **FP16→FP16**: 200-280 TFLOPS (current implementation)
- **FP16→FP32**: 150-250 TFLOPS (current implementation)
- **TF32→FP32**: 150-180 TFLOPS

### NVIDIA H100
- **FP16→FP16**: 400-600 TFLOPS (current implementation)
- **FP16→FP32**: 300-500 TFLOPS (current implementation)

*Note: PR #13511 aims to achieve 300+ TFLOPS on suitable hardware with optimizations.*

## Understanding Results

### TFLOPS (TeraFLOPS)
Operations per second for matrix multiplication: `2 × M × N × K / time`

### Arithmetic Intensity
FLOPS per byte transferred. Higher = more compute-bound (better for GPUs).
- Low (<10): Memory-bound - limited by bandwidth
- Medium (10-50): Balanced
- High (>50): Compute-bound - using GPU efficiently

### Bandwidth
Memory throughput in GB/s. Compare to GPU specs:
- RTX 4090: ~1000 GB/s
- A100: ~1500 GB/s
- H100: ~3000 GB/s

## Measurement Methodology

These benchmarks use the same timing method as tinygrad's official benchmarks:

1. **Create ExecItem**: Build the execution item from the scheduled kernel
2. **Direct kernel timing**: Use `ei.run(wait=True)` which returns GPU kernel time
3. **No Python overhead**: Measures actual GPU execution, not Python/scheduling overhead

This is more accurate than using Python's `time.perf_counter()` which includes:
- Python interpreter overhead
- Kernel launch overhead  
- CPU-GPU synchronization delays
- Other system latencies

Example from `triton_nv_matmul.py`:
```python
ei = ExecItem(CompiledRunner(prg), bufs, metadata)
tm = ei.run(wait=True)  # Returns GPU kernel time
tflops = (2*M*K*N/tm)*1e-12
```

## Debugging

If benchmark fails:
```bash
# Check CUDA is available
CUDA=1 python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# Verify GPU detection
CUDA=1 python3 -c "from tinygrad.runtime.ops_cuda import CUDADevice; d=CUDADevice('cuda:0'); print(d.arch)"

# Run with debug output
CUDA=1 DEBUG=2 python3 benchmark_matmul_simple.py 1024 1024 1024
```

## Comparing to Hand-Written CUDA

To compare with the hand-written CUDA version:
```bash
# Run hand-written benchmark
cd extra/gemm
CUDA=1 python3 cuda_matmul.py

# Compare with tinygrad's current implementation
cd ../..
CUDA=1 python3 benchmark_matmul_simple.py 4096 4096 4096 half
```

## Output Examples

### Simple Benchmark
```
Benchmarking 4096x4096x4096 matmul with dtypes.float16
Device: CUDA
  Iteration 1/10: 444.13 μs
  ...
Results:
  Min time:    444.13 μs
  Avg time:    456.82 μs
  Peak:        309.46 TFLOPS
  Average:     300.82 TFLOPS
  Bandwidth:   226.65 GB/s
```

### Comprehensive Benchmark
```
=== FP16 Input, FP32 Output (Typical ML Workload) ===
Benchmarking: M=4096, N=4096, K=4096
...
Peak TFLOPS:   245.32 TFLOPS
Mean TFLOPS:   241.18 TFLOPS
Bandwidth:     185.42 GB/s

SUMMARY TABLE
Size         In→Out       TC   Peak TFLOPS  Mean TFLOPS  BW (GB/s)
----------------------------------------------------------------------
4096x4096x4096 half→float  ON        245.32       241.18       185.42
```

