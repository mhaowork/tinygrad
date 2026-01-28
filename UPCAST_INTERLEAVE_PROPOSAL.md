# UPCAST: Sequential Load + Interleave vs Chunked Load

## The Key Insight

**Current UPCAST (Chunked)**: Load spatially separated chunks
**Better approach (Interleaved)**: Load sequential memory, distribute elements to different accumulators

## Current UPCAST=4 (Chunked - BAD)

### Memory Partitioning
```
Total elements: [0, 1, 2, 3, ..., 65535]

Partition into 4 CHUNKS:
  Chunk 0: [0..16383]       → Output[0], offset 0
  Chunk 1: [16384..32767]   → Output[1], offset 64KB
  Chunk 2: [32768..49151]   → Output[2], offset 128KB
  Chunk 3: [49152..65535]   → Output[3], offset 192KB
```

### Inner Loop
```c
for (int i = 0; i < 4096; i++) {
  // ❌ STRIDED LOADS - 64KB apart!
  float4 val0 = load(data + i*4 + 0);        // Chunk 0
  float4 val1 = load(data + i*4 + 16384);    // Chunk 1 (+64KB)
  float4 val2 = load(data + i*4 + 32768);    // Chunk 2 (+128KB)
  float4 val3 = load(data + i*4 + 49152);    // Chunk 3 (+192KB)

  // Shuffle into float16
  acc += (float16){val0[0..3], val1[0..3], val2[0..3], val3[0..3]};
}
```

**Problem**: Cache thrashing from 4 loads 64KB apart!

## Proposed UPCAST=4 (Interleaved - GOOD)

### Memory Partitioning
```
Load SEQUENTIALLY, distribute elements ROUND-ROBIN:

Sequential: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]

Distribute to accumulators:
  Acc 0: [0, 4, 8,  12, 16, 20, 24, 28, ...]  (every 4th starting at 0)
  Acc 1: [1, 5, 9,  13, 17, 21, 25, 29, ...]  (every 4th starting at 1)
  Acc 2: [2, 6, 10, 14, 18, 22, 26, 30, ...]  (every 4th starting at 2)
  Acc 3: [3, 7, 11, 15, 19, 23, 27, 31, ...]  (every 4th starting at 3)
```

### Inner Loop (Approach 1: Extract lanes)
```c
float4 acc0 = {0, 0, 0, 0};
float4 acc1 = {0, 0, 0, 0};
float4 acc2 = {0, 0, 0, 0};
float4 acc3 = {0, 0, 0, 0};

for (int i = 0; i < N; i += 16) {
  // ✅ SEQUENTIAL LOADS - all within 64 bytes!
  float4 val0 = load(data + i + 0);   // [0, 1, 2, 3]
  float4 val1 = load(data + i + 4);   // [4, 5, 6, 7]
  float4 val2 = load(data + i + 8);   // [8, 9, 10, 11]
  float4 val3 = load(data + i + 12);  // [12, 13, 14, 15]

  // Distribute elements to accumulators
  acc0 += (float4){val0[0], val1[0], val2[0], val3[0]};  // [0, 4, 8, 12]
  acc1 += (float4){val0[1], val1[1], val2[1], val3[1]};  // [1, 5, 9, 13]
  acc2 += (float4){val0[2], val1[2], val2[2], val3[2]};  // [2, 6, 10, 14]
  acc3 += (float4){val0[3], val1[3], val2[3], val3[3]};  // [3, 7, 11, 15]
}

// Final reduction
float4 result = acc0 + acc1 + acc2 + acc3;
```

**Benefits**:
- ✅ All loads within 64 bytes → **L1 cache hits**
- ✅ No cache thrashing
- ✅ Sequential memory access pattern
- ✅ Hardware prefetcher works perfectly

### Inner Loop (Approach 2: Transpose with vectors)
```c
float4 acc0 = {0, 0, 0, 0};
float4 acc1 = {0, 0, 0, 0};
float4 acc2 = {0, 0, 0, 0};
float4 acc3 = {0, 0, 0, 0};

for (int i = 0; i < N; i += 16) {
  // ✅ SEQUENTIAL LOADS
  float4 val0 = load(data + i + 0);   // [0, 1, 2, 3]
  float4 val1 = load(data + i + 4);   // [4, 5, 6, 7]
  float4 val2 = load(data + i + 8);   // [8, 9, 10, 11]
  float4 val3 = load(data + i + 12);  // [12, 13, 14, 15]

  // ✅ TRANSPOSE once (4 operations)
  // After transpose:
  //   tmp0 = [0, 4, 8, 12]
  //   tmp1 = [1, 5, 9, 13]
  //   tmp2 = [2, 6, 10, 14]
  //   tmp3 = [3, 7, 11, 15]
  float4 tmp0, tmp1, tmp2, tmp3;
  transpose_4x4(val0, val1, val2, val3, &tmp0, &tmp1, &tmp2, &tmp3);

  // ✅ Accumulate
  acc0 += tmp0;
  acc1 += tmp1;
  acc2 += tmp2;
  acc3 += tmp3;
}
```

**ARM NEON has efficient transpose**:
```asm
; Load 4 vectors sequentially
ldr   q0, [x1, #0]       ; val0
ldr   q1, [x1, #16]      ; val1
ldr   q2, [x1, #32]      ; val2
ldr   q3, [x1, #48]      ; val3

; Transpose using 4 instructions
zip1  v4.4s, v0.4s, v1.4s   ; [0, 4, 1, 5]
zip2  v5.4s, v0.4s, v1.4s   ; [2, 6, 3, 7]
zip1  v6.4s, v2.4s, v3.4s   ; [8, 12, 9, 13]
zip2  v7.4s, v2.4s, v3.4s   ; [10, 14, 11, 15]

zip1  v0.2d, v4.2d, v6.2d   ; [0, 4, 8, 12] → acc0
zip2  v1.2d, v4.2d, v6.2d   ; [1, 5, 9, 13] → acc1
zip1  v2.2d, v5.2d, v7.2d   ; [2, 6, 10, 14] → acc2
zip2  v3.2d, v5.2d, v7.2d   ; [3, 7, 11, 15] → acc3

; Accumulate
fadd  v16.4s, v16.4s, v0.4s
fadd  v17.4s, v17.4s, v1.4s
fadd  v18.4s, v18.4s, v2.4s
fadd  v19.4s, v19.4s, v3.4s
```

**Cost**: 6 transpose instructions **per iteration** in the loop
**But**: All loads are sequential → L1 cache hits!

## Performance Analysis

### Current UPCAST (Chunked)

**Per 16 elements**:
- 4 loads (4 cycles with L2 latency): **64 cycles** (16 cycles each × 4)
- 4 fadds (4 cycles): **4 cycles**
- 1 transpose after outer loop (amortized): **3 cycles**
- **Total: ~71 cycles per 16 elements**

**Memory bandwidth**: 34.2 GB/s (17% of peak) due to L2 hits

### Proposed UPCAST (Interleaved with transpose in loop)

**Per 16 elements**:
- 4 loads (4 cycles with L1 hits): **4 cycles** (1 cycle each)
- 6 transpose instructions: **6 cycles** (zip/unzip are 1 cycle on NEON)
- 4 fadds: **4 cycles**
- **Total: ~14 cycles per 16 elements**

**Memory bandwidth**: ~120 GB/s (60% of peak) due to L1 hits

**Speedup**: 71/14 = **5x faster!**

## Why Current UPCAST Doesn't Do This

The current UPCAST splits the **reduction axis** by creating separate output lanes that each handle a contiguous chunk of the data:

```python
# Conceptual UPCAST logic
for upcast_lane in range(4):
    offset = upcast_lane * (N / 4)
    for i in range(N / 4):
        output[upcast_lane] += data[offset + i]
```

This is designed for operations where **each output lane processes independent data** (like different output positions in matrix multiply).

**For pure reductions**, this creates the strided access problem!

## The Fix

Modify UPCAST codegen for CPU reductions to use **interleaved** access pattern:

```python
# Interleaved access pattern
for i in range(N / 4):
    # Load 4 consecutive elements
    val = load_sequential(data + i * 4, count=4)

    # Distribute to output lanes (transpose)
    for lane in range(4):
        output[lane] += val[lane]
```

**In generated code**:
```c
// Instead of:
float4 val0 = load(data + i*4 + 0);
float4 val1 = load(data + i*4 + 16384);  // +64KB
float4 val2 = load(data + i*4 + 32768);  // +128KB
float4 val3 = load(data + i*4 + 49152);  // +192KB

// Generate:
float4 v0 = load(data + i*16 + 0);   // Sequential
float4 v1 = load(data + i*16 + 4);   // Sequential
float4 v2 = load(data + i*16 + 8);   // Sequential
float4 v3 = load(data + i*16 + 12);  // Sequential
transpose_4x4(&v0, &v1, &v2, &v3);
acc0 += v0;
acc1 += v1;
acc2 += v2;
acc3 += v3;
```

## Trade-off Analysis

### Current (Chunked)
- ❌ Strided loads → L2 cache → 64 cycle latency
- ✅ No transpose in loop
- ✅ Simple addressing
- **Result**: 71 cycles/16elem, 34 GB/s (17% bandwidth)

### Proposed (Interleaved)
- ✅ Sequential loads → L1 cache → 4 cycle latency
- ❌ Transpose in loop (6 cycles)
- ❌ More complex codegen
- **Result**: 14 cycles/16elem, 120 GB/s (60% bandwidth)

**The transpose overhead (6 cycles) is MUCH cheaper than L2 cache misses (60 cycles)!**

## Implementation Path

### Option 1: Modify UPCAST codegen for CPU reductions
When generating UPCAST for pure reductions on CPU:
1. Detect: `is_cpu_reduce = device == "CPU" and reduceop is not None`
2. Generate interleaved loads instead of chunked loads
3. Insert transpose before accumulation
4. Keep same output structure (vec(4))

### Option 2: New optimization pass
Add a new optimization that:
1. Detects UPCAST on reductions
2. Transforms loads from chunked to interleaved
3. Inserts transpose UOps

### Option 3: Pattern in devectorizer
In devectorizer, detect the strided UPCAST pattern and transform it to interleaved.

## Expected Performance

With interleaved UPCAST:
- **2048x2048**: 14 cycles/16elem vs 71 cycles/16elem = **5x faster**
- This would make tinygrad **3.5x faster than current** (from 1.54x slower to ~0.44x faster)
- **Potentially faster than PyTorch** if we optimize the transpose well

## Why This is Better Than Disabling UPCAST

**Disabling UPCAST**: Still 1.5x slower, 32 outer loops
**Interleaved UPCAST**: Potentially faster than PyTorch, 8 outer loops, vectorized throughout

This keeps all the benefits of UPCAST (fewer outer loops, vectorization) while eliminating the cache thrashing!
