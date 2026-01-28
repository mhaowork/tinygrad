import torch
import ctypes
import subprocess
import tempfile
import os

# Force single thread to isolate the kernel
torch.set_num_threads(1)

# Create test tensor
x = torch.randn(4096, 4096, dtype=torch.float32)

# Warmup
_ = x.sum()

# Try to find PyTorch's native function
# PyTorch uses ATen which has optimized kernels
print("PyTorch version:", torch.__version__)
print("PyTorch file:", torch.__file__)

# Let's create a simple profiling script to capture what PyTorch is doing
print("\n=== Testing PyTorch sum performance ===")
import time

# Small benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    result = x.sum()
    end = time.perf_counter()
    times.append((end - start) * 1000)

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.3f} ms")
print(f"Result: {result.item():.2f}")

# Calculate bandwidth
data_size = 4096 * 4096 * 4  # bytes
bandwidth = (data_size / (avg_time / 1000)) / 1e9  # GB/s
print(f"Bandwidth: {bandwidth:.1f} GB/s")

# Try to find the actual implementation
# PyTorch's sum is likely implemented in native code
try:
    # Check if we can get function pointer
    print("\n=== Trying to locate PyTorch's sum implementation ===")

    # On macOS with Apple Silicon, PyTorch uses Accelerate framework or its own kernels
    # Let's check what libraries PyTorch loads
    import sys
    if sys.platform == "darwin":
        print("\nChecking loaded libraries with otool...")
        torch_lib = torch.__file__.replace("__init__.py", "lib/libtorch_cpu.dylib")
        if os.path.exists(torch_lib):
            result = subprocess.run(
                ["otool", "-L", torch_lib],
                capture_output=True,
                text=True
            )
            print(result.stdout)

            # Check if it uses Accelerate
            if "Accelerate" in result.stdout:
                print("\n✓ PyTorch is using Apple's Accelerate framework!")
                print("  This includes highly optimized BLAS/LAPACK routines")

            if "vecLib" in result.stdout:
                print("\n✓ PyTorch is using vecLib (part of Accelerate)")

except Exception as e:
    print(f"Could not inspect libraries: {e}")

# Let's write a simple C test using the same approach as PyTorch might
print("\n=== Creating test with different optimization strategies ===")

test_code = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

// Strategy 1: Simple loop with vector accumulator (like tinygrad)
float reduce_simple(float* data, int size) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        acc = vaddq_f32(acc, v);
    }
    float result = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                   vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
    return result;
}

// Strategy 2: Multiple accumulators to hide latency
float reduce_multi_acc(float* data, int size) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < size; i += 16) {
        acc0 = vaddq_f32(acc0, vld1q_f32(&data[i+0]));
        acc1 = vaddq_f32(acc1, vld1q_f32(&data[i+4]));
        acc2 = vaddq_f32(acc2, vld1q_f32(&data[i+8]));
        acc3 = vaddq_f32(acc3, vld1q_f32(&data[i+12]));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    return vaddvq_f32(acc0);  // Horizontal add (ARMv8)
}

// Strategy 3: Unrolled with prefetch hints
float reduce_prefetch(float* data, int size) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < size; i += 16) {
        __builtin_prefetch(&data[i + 64], 0, 0);
        acc0 = vaddq_f32(acc0, vld1q_f32(&data[i+0]));
        acc1 = vaddq_f32(acc1, vld1q_f32(&data[i+4]));
        acc2 = vaddq_f32(acc2, vld1q_f32(&data[i+8]));
        acc3 = vaddq_f32(acc3, vld1q_f32(&data[i+12]));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    return vaddvq_f32(acc0);
}

int main() {
    int size = 4096 * 4096;
    float* data = aligned_alloc(64, size * sizeof(float));

    // Initialize with random data
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    clock_t start, end;
    double cpu_time;

    // Warmup
    volatile float result;
    result = reduce_simple(data, size);

    // Benchmark simple
    start = clock();
    for (int i = 0; i < 100; i++) {
        result = reduce_simple(data, size);
    }
    end = clock();
    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC * 10;
    printf("Simple:    %.3f ms (%.1f GB/s)\\n", cpu_time,
           (size * 4 / (cpu_time / 1000)) / 1e9);

    // Benchmark multi-accumulator
    start = clock();
    for (int i = 0; i < 100; i++) {
        result = reduce_multi_acc(data, size);
    }
    end = clock();
    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC * 10;
    printf("Multi-acc: %.3f ms (%.1f GB/s)\\n", cpu_time,
           (size * 4 / (cpu_time / 1000)) / 1e9);

    // Benchmark with prefetch
    start = clock();
    for (int i = 0; i < 100; i++) {
        result = reduce_prefetch(data, size);
    }
    end = clock();
    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC * 10;
    printf("Prefetch:  %.3f ms (%.1f GB/s)\\n", cpu_time,
           (size * 4 / (cpu_time / 1000)) / 1e9);

    free(data);
    return 0;
}
"""

# Write and compile the test
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(test_code)
    c_file = f.name

try:
    exe_file = c_file.replace('.c', '')
    compile_cmd = ['clang', '-O3', '-march=native', c_file, '-o', exe_file]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\n=== Running C optimization tests ===")
        result = subprocess.run([exe_file], capture_output=True, text=True)
        print(result.stdout)
        os.unlink(exe_file)
    else:
        print("Compilation failed:", result.stderr)

finally:
    os.unlink(c_file)

print("\n=== Summary ===")
print(f"PyTorch (1 thread): {avg_time:.3f} ms, {bandwidth:.1f} GB/s")
print(f"Tinygrad:          ~2.35 ms, ~29 GB/s")
