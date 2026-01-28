"""
Memory-only benchmark to measure pure memory load bandwidth.
No arithmetic operations - just loads.

This is a reference to compare against actual reduce operations.
The goal is to understand how much of the performance gap is due to
memory bandwidth vs. compute overhead.

Usage:
  python extra/mem_only_reduce_speed.py
"""
import numpy as np
import ctypes
import time
from tinygrad import Tensor, GlobalCounters, Context, Device
from tinygrad.engine.realize import CompiledRunner
from dataclasses import replace

# =============================================================================
# VERSION 1: Memory-only (no math) - measures pure memory bandwidth
# =============================================================================
mem_only_src = """
// data1 is 16M inputs - MEMORY ONLY (no add operations)
// This measures pure memory load bandwidth
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 val0, val1, val2, val3;
  
  // Just load data, no accumulation
  for (int ridx0 = 0; ridx0 < 16777216; ridx0 += 16) {
    val0 = *((float4*)((data1 + ridx0 + 0)));
    val1 = *((float4*)((data1 + ridx0 + 4)));
    val2 = *((float4*)((data1 + ridx0 + 8)));
    val3 = *((float4*)((data1 + ridx0 + 12)));
  }
  
  // Prevent compiler from optimizing away loads - use last value
  *(data0 + 0) = val3[0];
}
"""

# =============================================================================
# VERSION 2: Memory-only with volatile to prevent optimization
# =============================================================================
mem_only_volatile_src = """
// data1 is 16M inputs - MEMORY ONLY with volatile
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));

void reduce(float* restrict data0, float* restrict data1) {
  volatile float4 val0, val1, val2, val3;
  
  for (int ridx0 = 0; ridx0 < 16777216; ridx0 += 16) {
    val0 = *((float4*)((data1 + ridx0 + 0)));
    val1 = *((float4*)((data1 + ridx0 + 4)));
    val2 = *((float4*)((data1 + ridx0 + 8)));
    val3 = *((float4*)((data1 + ridx0 + 12)));
  }
  
  *(data0 + 0) = val3[0];
}
"""

# =============================================================================
# VERSION 3: Actual reduce with vector accumulators (from image - 74 GB/s)
# Two float4 accumulators, 4 loads per iteration
# =============================================================================
reduce_with_accum_src = """
// data1 is 16M inputs
// 74 GB/s version with 2 vector accumulators
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  
  for (int ridx0 = 0; ridx0 < 16777216; ridx0 += 16) {
    float4 val0 = *((float4*)((data1 + ridx0 + 0)));
    float4 val1 = *((float4*)((data1 + ridx0 + 4)));
    float4 val2 = *((float4*)((data1 + ridx0 + 8)));
    float4 val3 = *((float4*)((data1 + ridx0 + 12)));
    acc0 += val0 + val1;  // Vector add
    acc1 += val2 + val3;  // Vector add
  }
  
  // Horizontal sum at the end
  *(data0 + 0) = acc0[0] + acc0[1] + acc0[2] + acc0[3] + 
                 acc1[0] + acc1[1] + acc1[2] + acc1[3];
}
"""

# =============================================================================
# VERSION 4: Scalar accumulator (SLOW - what tinygrad was generating before)
# =============================================================================
reduce_scalar_accum_src = """
// data1 is 16M inputs
// SLOW version with scalar accumulator - horizontal sum INSIDE loop
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));

void reduce(float* restrict data0, float* restrict data1) {
  float acc0 = 0.0f;  // Scalar accumulator!
  
  for (int ridx0 = 0; ridx0 < 16777216; ridx0 += 4) {
    float4 val0 = *((float4*)((data1 + ridx0)));
    // BAD: Horizontal sum inside loop!
    acc0 += val0[0] + val0[1] + val0[2] + val0[3];
  }
  
  *(data0 + 0) = acc0;
}
"""

# =============================================================================
# VERSION 5: Vector accumulator (FAST - what tinygrad generates after our fix)
# =============================================================================
reduce_vector_accum_src = """
// data1 is 16M inputs
// FAST version with vector accumulator - horizontal sum at END
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};  // Vector accumulator!
  
  for (int ridx0 = 0; ridx0 < 16777216; ridx0 += 4) {
    float4 val0 = *((float4*)((data1 + ridx0)));
    acc0 += val0;  // Vector add - single SIMD instruction!
  }
  
  // Horizontal sum ONLY at the end
  *(data0 + 0) = acc0[0] + acc0[1] + acc0[2] + acc0[3];
}
"""

# =============================================================================
# VERSION 6: 8 accumulators (maximum parallelism - from reduce_speed.py)
# =============================================================================
reduce_8_accum_src = """
// data1 is 16M inputs
// Maximum parallelism with 8 vector accumulators
typedef float float4 __attribute__((aligned(32),vector_size(16)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc4 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc5 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc6 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc7 = {0.0f, 0.0f, 0.0f, 0.0f};
  
  float* data1_1 = data1 + 4194304;
  float* data1_2 = data1 + (4194304 * 2);
  float* data1_3 = data1 + (4194304 * 3);
  
  for (int ridx0 = 0; ridx0 < 16777216/4; ridx0 += 16) {
    float4 val0 = *(float4*)((data1 + (ridx0 + 0)));
    float4 val1 = *(float4*)((data1 + (ridx0 + 4)));
    float4 val2 = *(float4*)((data1 + (ridx0 + 8)));
    float4 val3 = *(float4*)((data1 + (ridx0 + 12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    
    val0 = *(float4*)((data1_1 + (ridx0 + 0)));
    val1 = *(float4*)((data1_1 + (ridx0 + 4)));
    val2 = *(float4*)((data1_1 + (ridx0 + 8)));
    val3 = *(float4*)((data1_1 + (ridx0 + 12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
    
    val0 = *(float4*)((data1_2 + (ridx0 + 0)));
    val1 = *(float4*)((data1_2 + (ridx0 + 4)));
    val2 = *(float4*)((data1_2 + (ridx0 + 8)));
    val3 = *(float4*)((data1_2 + (ridx0 + 12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    
    val0 = *(float4*)((data1_3 + (ridx0 + 0)));
    val1 = *(float4*)((data1_3 + (ridx0 + 4)));
    val2 = *(float4*)((data1_3 + (ridx0 + 8)));
    val3 = *(float4*)((data1_3 + (ridx0 + 12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
  }
  
  float4 out = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  *(data0 + 0) = out[0] + out[1] + out[2] + out[3];
}
"""

VERSIONS = {
    "mem_only": (mem_only_src, "Memory-only (no math)"),
    "mem_only_volatile": (mem_only_volatile_src, "Memory-only with volatile"),
    "scalar_accum": (reduce_scalar_accum_src, "Scalar accumulator (SLOW)"),
    "vector_accum": (reduce_vector_accum_src, "Vector accumulator (FAST)"),
    "2_accum": (reduce_with_accum_src, "2 vector accumulators"),
    "8_accum": (reduce_8_accum_src, "8 vector accumulators (max)"),
}

def run_benchmark(src: str, name: str, np_array: np.ndarray, verify: bool = True):
    """Run a single benchmark with given C source code."""
    a = Tensor(np_array).realize()

    with Context(SPLIT_REDUCEOP=0):
        GlobalCounters.reset()
        out = a.sum()
        schedule = out.schedule()
        for i, ei in enumerate(schedule):
            ei.lower()
            if i == 0:
                prg_spec = ei.prg.p
                prg_spec = replace(prg_spec, name="reduce", src=src, lib=None)
                prg = CompiledRunner(prg_spec)
                schedule[i] = replace(ei, prg=prg)

        st = time.perf_counter()
        for ei in schedule:
            ei.run()
        Device[a.device].synchronize()
        et = time.perf_counter()

        result = out.item()

        if verify and "mem_only" not in name:
            expected = np_array.sum()
            np.testing.assert_allclose(result, expected, atol=1, rtol=1e-4)

        elapsed_s = et - st
        gbps = (np_array.nbytes / 1e9) / elapsed_s if elapsed_s > 0 else float("inf")
        print(f"  time: {elapsed_s*1e3:8.3f} ms  bandwidth: {gbps:6.2f} GB/s")
        return result

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Memory Bandwidth Benchmark: Pure Loads vs Reduce Operations")
    print("=" * 70)
    print(f"Data size: 4096 x 4096 = 16M floats = 64 MB")
    print()
    
    # Create test data
    np_array = (np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5)
    
    # Run selected version or all versions
    if len(sys.argv) > 1:
        version = sys.argv[1]
        if version in VERSIONS:
            src, desc = VERSIONS[version]
            print(f"\nRunning: {desc}")
            print("-" * 50)
            run_benchmark(src, version, np_array, verify=(version != "mem_only" and version != "mem_only_volatile"))
        else:
            print(f"Unknown version: {version}")
            print(f"Available: {list(VERSIONS.keys())}")
    else:
        # Run all versions for comparison
        print("\nRunning all versions for comparison...")
        print("-" * 70)
        
        for version, (src, desc) in VERSIONS.items():
            print(f"\n{desc}:")
            try:
                run_benchmark(src, version, np_array, verify=("mem_only" not in version))
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n" + "=" * 70)
        print("Key insights:")
        print("  - mem_only: Shows theoretical max bandwidth (no compute)")
        print("  - scalar_accum: What tinygrad generated BEFORE our fix")
        print("  - vector_accum: What tinygrad generates AFTER our fix")
        print("  - 8_accum: Maximum parallelism (manual optimization)")
        print("=" * 70)
