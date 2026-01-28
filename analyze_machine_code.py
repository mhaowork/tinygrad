#!/usr/bin/env python3
"""First principles analysis of tinygrad's machine code"""

import numpy as np

print("="*80)
print("MACHINE CODE ANALYSIS - FIRST PRINCIPLES")
print("="*80)

# Problem size
SIZE = 2048
ELEMENTS = SIZE * SIZE
BYTES = ELEMENTS * 4  # float32

print(f"\nProblem: Sum of {SIZE}x{SIZE} = {ELEMENTS:,} elements ({BYTES/1024/1024:.1f} MB)")

# From our assembly analysis
print("\n" + "="*80)
print("TINYGRAD KERNEL STRUCTURE (from DEBUG=7 output)")
print("="*80)

print("""
void r_8_8_4_4096_4(float* data0_256, float* data1_4194304, int core_id) {
  float16 acc0[1];
  int gidx0 = core_id; /* 8 threads */

  for (int Lidx2 = 0; Lidx2 < 8; Lidx2++) {           // OUTER: 8 iterations per thread
    *(acc0+0) = (float16){0,...,0};                   // Reset accumulator

    for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {      // INNER: 4096 iterations
      int alu1 = ((gidx0<<19)+(Lidx2<<16)+(Ridx0<<2));

      // Load 4 float4 vectors from STRIDED offsets
      float4 val0 = load(data + (alu1 + 16384));      // Offset +16384
      float4 val1 = load(data + (alu1 + 32768));      // Offset +32768
      float4 val2 = load(data + (alu1 + 49152));      // Offset +49152
      float4 val3 = load(data + (alu1 + 0));          // Offset +0

      // Shuffle into float16 and accumulate
      *(acc0+0) += (float16){val3[0],val3[1],val3[2],val3[3],
                             val0[0],val0[1],val0[2],val0[3],
                             val1[0],val1[1],val1[2],val1[3],
                             val2[0],val2[1],val2[2],val2[3]};
    }

    // TRANSPOSE and horizontal reduction
    store = (float4){acc[0],acc[4],acc[8],acc[12]} +
            (float4){acc[1],acc[5],acc[9],acc[13]} +
            (float4){acc[2],acc[6],acc[10],acc[14]} +
            (float4){acc[3],acc[7],acc[11],acc[15]};
  }
}
""")

print("\n" + "="*80)
print("INSTRUCTION COUNT PER INNER LOOP ITERATION")
print("="*80)

inner_loop_instructions = {
    "Address calculation (add, lsl)": 3,
    "Loads (ldr q)": 4,
    "Vector adds (fadd v.4s)": 4,
    "Loop control (add, cmp, b.ne)": 3,
}

total_per_iter = sum(inner_loop_instructions.values())

print("Per iteration:")
for name, count in inner_loop_instructions.items():
    print(f"  {name:35s}: {count:2d} instructions")
print(f"  {'TOTAL':35s}: {total_per_iter:2d} instructions")

iterations_per_thread = 8 * 4096  # outer * inner
total_instructions_inner = iterations_per_thread * total_per_iter * 8  # 8 threads

print(f"\nInner loop:")
print(f"  Iterations per thread: {iterations_per_thread:,}")
print(f"  Instructions per thread: {iterations_per_thread * total_per_iter:,}")
print(f"  Total (8 threads): {total_instructions_inner:,}")

print("\n" + "="*80)
print("TRANSPOSE OVERHEAD (after each outer loop)")
print("="*80)

transpose_instructions = {
    "zip1": 2,
    "ext": 2,
    "mov": 2,
    "trn2": 2,
    "fadd": 4,
    "str": 1,
}

total_transpose = sum(transpose_instructions.values())
print("Per outer loop iteration:")
for name, count in transpose_instructions.items():
    print(f"  {name:35s}: {count:2d} instructions")
print(f"  {'TOTAL':35s}: {total_transpose:2d} instructions")

transpose_total = total_transpose * 8 * 8  # 8 outer loops * 8 threads
print(f"\nTotal transpose overhead (8 outer × 8 threads): {transpose_total:,} instructions")

print("\n" + "="*80)
print("MEMORY ACCESS PATTERN")
print("="*80)

print("\nPer inner loop iteration:")
print(f"  Loads: 4 × float4 = 4 × 16 bytes = 64 bytes")
print(f"  Store: 1 × float4 = 16 bytes (after outer loop)")

loads_per_thread = 8 * 4096 * 64
stores_per_thread = 8 * 16
total_memory_per_thread = loads_per_thread + stores_per_thread
total_memory = total_memory_per_thread * 8

print(f"\nPer thread:")
print(f"  Loads:  {loads_per_thread:,} bytes ({loads_per_thread/1024/1024:.1f} MB)")
print(f"  Stores: {stores_per_thread:,} bytes ({stores_per_thread/1024:.1f} KB)")
print(f"  Total:  {total_memory_per_thread:,} bytes ({total_memory_per_thread/1024/1024:.1f} MB)")

print(f"\nAll threads:")
print(f"  Total memory traffic: {total_memory:,} bytes ({total_memory/1024/1024:.1f} MB)")
print(f"  Input data size: {BYTES:,} bytes ({BYTES/1024/1024:.1f} MB)")
print(f"  Ratio: {total_memory/BYTES:.2f}x (should be ~1.0 for optimal)")

print("\n" + "="*80)
print("CACHE ANALYSIS - STRIDE PATTERN")
print("="*80)

print("""
For thread 0, outer loop iteration 0, the loads are:
  Ridx0=0: load(0 + 0),      load(0 + 16384), load(0 + 32768), load(0 + 49152)
  Ridx0=1: load(4 + 0),      load(4 + 16384), load(4 + 32768), load(4 + 49152)
  Ridx0=2: load(8 + 0),      load(8 + 16384), load(8 + 32768), load(8 + 49152)
  ...

Offsets in bytes: 0, 65536, 131072, 196608 (each 64KB apart)

M1/M2 L1 cache: 128-192 KB data cache
Cache line: 64 bytes

PROBLEM: Loading from 4 locations 64KB apart causes:
  - 4 separate cache line fetches per iteration
  - Likely L1 cache thrashing (4×64KB = 256KB > L1 size)
  - Each load misses L1, goes to L2 or memory
""")

print("\n" + "="*80)
print("THEORETICAL PERFORMANCE")
print("="*80)

# M1/M2 specs (approximate)
CPU_FREQ_GHZ = 3.2  # M1/M2 performance cores
NEON_PIPES = 2  # Can issue 2 NEON ops per cycle
LOAD_LATENCY = 4  # L1 cache hit latency
LOAD_THROUGHPUT = 2  # Can do 2 loads per cycle
FADD_LATENCY = 2  # FADD latency
FADD_THROUGHPUT = 2  # Can do 2 FADD per cycle

print(f"\nApple M1/M2 specs:")
print(f"  CPU frequency: {CPU_FREQ_GHZ} GHz")
print(f"  NEON pipes: {NEON_PIPES}")
print(f"  Load throughput: {LOAD_THROUGHPUT} per cycle")
print(f"  FADD throughput: {FADD_THROUGHPUT} per cycle")

print(f"\nInner loop bottleneck analysis:")
print(f"  4 loads: 4 / {LOAD_THROUGHPUT} = 2 cycles (throughput bound)")
print(f"  4 fadds: 4 / {FADD_THROUGHPUT} = 2 cycles (throughput bound)")
print(f"  Loop overhead: ~1 cycle")
print(f"  Best case: 5 cycles per iteration")

best_cycles_per_iter = 5
best_cycles_total = iterations_per_thread * best_cycles_per_iter
best_time_ns = best_cycles_total / (CPU_FREQ_GHZ * 1e9) * 1e9
best_time_ms = best_time_ns / 1e6

print(f"\nBest case (L1 cache, perfect pipelining):")
print(f"  Cycles per thread: {best_cycles_total:,}")
print(f"  Time per thread: {best_time_ms:.3f} ms")
print(f"  With 8 threads (parallel): {best_time_ms:.3f} ms")

# With cache misses
L2_LATENCY = 20  # L2 hit latency in cycles
miss_prob = 0.75  # Assume 75% L1 miss rate due to striding
avg_load_latency = LOAD_LATENCY * (1 - miss_prob) + L2_LATENCY * miss_prob

worst_cycles_per_iter = 4 * avg_load_latency + 4 + 1  # loads + fadds + overhead
worst_cycles_total = iterations_per_thread * worst_cycles_per_iter
worst_time_ms = worst_cycles_total / (CPU_FREQ_GHZ * 1e9) * 1000

print(f"\nWith cache misses (75% L1 miss → L2):")
print(f"  Avg load latency: {avg_load_latency:.1f} cycles")
print(f"  Cycles per iteration: {worst_cycles_per_iter:.1f}")
print(f"  Cycles per thread: {worst_cycles_total:,.0f}")
print(f"  Time per thread: {worst_time_ms:.3f} ms")
print(f"  With 8 threads (parallel): {worst_time_ms:.3f} ms")

print("\n" + "="*80)
print("COMPARISON WITH MEASURED PERFORMANCE")
print("="*80)

measured_tinygrad = 0.49  # ms from benchmark
measured_pytorch_8t = 0.14  # ms from benchmark
measured_vdsp_1t = 0.30  # ms from benchmark

print(f"\nMeasured performance:")
print(f"  tinygrad (8 threads):  {measured_tinygrad:.2f} ms")
print(f"  PyTorch (8 threads):   {measured_pytorch_8t:.2f} ms")
print(f"  vDSP (1 thread):       {measured_vdsp_1t:.2f} ms")

print(f"\nTheoretical vs measured:")
print(f"  Best case (L1 hit):    {best_time_ms:.2f} ms  ({measured_tinygrad/best_time_ms:.1f}x slower than theory)")
print(f"  With L2 misses:        {worst_time_ms:.2f} ms  ({measured_tinygrad/worst_time_ms:.1f}x of theory)")

print(f"\nConclusion:")
print(f"  tinygrad is close to theoretical with L2 misses")
print(f"  The STRIDED memory pattern is likely the main bottleneck")
print(f"  PyTorch is {measured_tinygrad/measured_pytorch_8t:.1f}x faster → must have better memory access pattern")

print("\n" + "="*80)
print("MEMORY BANDWIDTH ANALYSIS")
print("="*80)

# M1/M2 memory bandwidth
MEMORY_BW_GBs = 200  # GB/s unified memory bandwidth

data_read = BYTES
time_s = measured_tinygrad / 1000
achieved_bw = data_read / time_s / 1e9

print(f"\nM1/M2 unified memory bandwidth: ~{MEMORY_BW_GBs} GB/s")
print(f"Data read: {data_read/1024/1024:.1f} MB")
print(f"Time: {measured_tinygrad:.2f} ms")
print(f"Achieved bandwidth: {achieved_bw:.1f} GB/s ({achieved_bw/MEMORY_BW_GBs*100:.1f}% of peak)")

pytorch_bw = data_read / (measured_pytorch_8t / 1000) / 1e9
print(f"\nPyTorch achieved bandwidth: {pytorch_bw:.1f} GB/s ({pytorch_bw/MEMORY_BW_GBs*100:.1f}% of peak)")

print(f"\nConclusion:")
print(f"  tinygrad achieves {achieved_bw/MEMORY_BW_GBs*100:.1f}% of memory bandwidth")
print(f"  PyTorch achieves {pytorch_bw/MEMORY_BW_GBs*100:.1f}% of memory bandwidth")
print(f"  Both are memory-bandwidth bound (good)")
print(f"  BUT: tinygrad's strided access pattern is likely causing cache thrashing")
print(f"  This explains the {measured_tinygrad/measured_pytorch_8t:.1f}x slowdown")
