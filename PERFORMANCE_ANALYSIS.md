# Performance Analysis: Tinygrad CPU Sum Performance

## Summary (CORRECTED)
**Threading is ESSENTIAL and working well!**

With threading (HEAD commit 374ac3dc6):
- ✅ **4096x4096**: Tinygrad **0.95x faster than PyTorch** (1.17ms vs 1.23ms)
- ⚠️ **2048x2048**: Tinygrad **1.33x slower than PyTorch** (0.47ms vs 0.35ms)

Without threading (THREADS=0):
- ❌ **4096x4096**: Tinygrad **3.29x slower than PyTorch** (4.30ms vs 1.31ms)
- ❌ **2048x2048**: Tinygrad **3.39x slower than PyTorch** (1.15ms vs 0.34ms)

## Benchmark Results (Proper Test Using external_test_speed_v_torch.py)

### With Threading (Default, THREADS=1)

| Size | PyTorch | Tinygrad | Result |
|------|---------|----------|---------|
| 2048x2048 | 0.35 ms | 0.47 ms | **1.33x slower** ⚠️ |
| 4096x4096 | 1.23 ms | 1.17 ms | **0.95x faster** ✅ |

### Without Threading (THREADS=0)

| Size | PyTorch | Tinygrad | Result |
|------|---------|----------|---------|
| 2048x2048 | 0.34 ms | 1.15 ms | **3.39x slower** ❌ |
| 4096x4096 | 1.31 ms | 4.30 ms | **3.29x slower** ❌ |

**Key Finding**: Threading is critical! It provides **3.5x speedup** and makes tinygrad competitive with PyTorch for large arrays.

### Beam Search Results (BEAM=2)

| Size | PyTorch | Tinygrad (BEAM=0) | Tinygrad (BEAM=2) | Improvement |
|------|---------|-------------------|-------------------|-------------|
| 2048x2048 | 0.35 ms | 0.47 ms (1.33x slower) | 0.53 ms (1.53x slower) | **Worse** ❌ |
| 4096x4096 | 1.28 ms | 1.17 ms (0.95x faster) | 0.92 ms (0.72x faster) | **Better** ✅ |

Beam search helps 4096x4096 but hurts 2048x2048!

## Recent Changes Analysis

### HEAD: Threading Support (commit 374ac3dc6)
**Changes**: Added multi-threading to LLVM CPU renderer
- `has_threads = bool(getenv("THREADS", 1))` - enabled by default
- `global_max = (CPU_COUNT.value, 0, 0)` - splits work across cores
- Added `core_id` parameter to kernels

**Status**: ✅ **Threading is working** - kernel shows `gidx0 = core_id; /* 8 */`

### HEAD~1: Devectorizer Fix (commit 4a7648b9c)
**Changes**: Prevents devectorizing SIMD ADD operations on CPU
- Added check to skip devectorizing float ADD ops on CPU
- Modified `reduce_to_acc` to use vector accumulators (`vector_acc` path)

**Status**: ⚠️ **Partially working** - only applies to simple reductions

## Root Cause Analysis

### Why 4096x4096 is Fast (0.95x faster than PyTorch)

The 4096x4096 kernel gets **3 optimizations** applied:
```
Opt(op=OptOps.UPCAST, axis=0, arg=4)    ← Extra optimization!
Opt(op=OptOps.UNROLL, axis=0, arg=4)
Opt(op=OptOps.THREAD, axis=0, arg=8)
```

This generates 3 reduction kernels with good threading and vectorization.

### Why 2048x2048 is Slower (1.33x slower than PyTorch)

The 2048x2048 kernel only gets **2 optimizations**:
```
Opt(op=OptOps.UNROLL, axis=0, arg=4)
Opt(op=OptOps.THREAD, axis=0, arg=8)
```

**Missing**: `UPCAST` optimization that 4096x4096 gets!

This generates only 2 reduction kernels:
- `r_32_8_4096_4` (main reduction, 0.87ms)
- `r_64_4` (final reduction, ~0.01ms)

vs 4096x4096's 3 kernels with better work distribution.

### Additional Issues (from earlier analysis)

### Issue 1: SPLIT_REDUCEOP Defeats Vectorization

With `SPLIT_REDUCEOP=1` (default), the sum is split into 3 kernels. The first kernel shows:

```c
for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
  float4 val0 = (*((float4*)((data1_16777216+(alu4+4096)))));
  float4 val1 = (*((float4*)((data1_16777216+(alu4+8192)))));
  float4 val2 = (*((float4*)((data1_16777216+(alu4+12288)))));
  float4 val3 = (*((float4*)((data1_16777216+alu4))));
  *(acc0+0) = ((*(acc0+0))+val3[0]+val3[1]+val3[2]+val3[3]);  // ❌ Horizontal reduction!
  *(acc0+1) = ((*(acc0+1))+val0[0]+val0[1]+val0[2]+val0[3]);  // ❌ Scalar adds
  *(acc0+2) = ((*(acc0+2))+val1[0]+val1[1]+val1[2]+val1[3]);  // ❌ No vectorization
  *(acc0+3) = ((*(acc0+3))+val2[0]+val2[1]+val2[2]+val2[3]);  // ❌ Dependency chains
}
```

**Problem**: Loads `float4` vectors but immediately does horizontal reduction (`val3[0]+val3[1]+val3[2]+val3[3]`), resulting in scalar additions instead of vector operations.

### Issue 2: Vector Accumulator Fix Doesn't Apply to Split Reductions

The `vector_acc` optimization in [devectorizer.py:301-324](tinygrad/codegen/late/devectorizer.py#L301-L324) only works for simple reductions. When `SPLIT_REDUCEOP` splits the reduction into multiple kernels, the intermediate reductions don't use vector accumulators.

### Issue 3: Comparison with Non-Split Reduction

With `SPLIT_REDUCEOP=0` (from baseline_reduce_benchmark.py), the kernel is better:

```c
void r_4194304_4(float* restrict data0_1, float* restrict data1_16777216, int core_id) {
  float acc0[4];
  *(acc0+0) = 0.0f;
  *(acc0+1) = 0.0f;
  *(acc0+2) = 0.0f;
  *(acc0+3) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 4194304; Ridx0++) {
    float4 val0 = (*((float4*)((data1_16777216+(Ridx0<<2)))));
    *(acc0+0) = ((*(acc0+0))+val0[0]);  // Still scalar, but simpler
    *(acc0+1) = ((*(acc0+1))+val0[1]);
    *(acc0+2) = ((*(acc0+2))+val0[2]);
    *(acc0+3) = ((*(acc0+3))+val0[3]);
  }
  *(data0_1+0) = ((*(acc0+0))+(*(acc0+1))+(*(acc0+2))+(*(acc0+3)));
}
```

**Assembly output** (the C gets compiled to):
```asm
0x000004: movi	v0.2d, #0000000000000000  ; acc = 0
0x000008: ldr	q1, [x1, x8, lsl #4]      ; load float4
0x00000c: fadd	v0.4s, v0.4s, v1.4s      ; ✅ Vector ADD!
```

LLVM optimizes the 4 scalar accumulators into a single vector register, producing good code. But performance is still poor (~3.9ms).

### Issue 4: Single Vector Accumulator Creates Dependency Chain

The assembly shows only **one vector accumulator** (v0), creating a dependency chain:
```asm
fadd v0.4s, v0.4s, v1.4s  ; v0 depends on previous v0
fadd v0.4s, v0.4s, v1.4s  ; can't execute until previous fadd completes
```

On modern CPUs with multiple execution units, we need **multiple independent accumulators** for instruction-level parallelism (ILP).

## What PyTorch/Accelerate Does Better

1. **Multiple vector accumulators** (4-8 independent accumulators) for ILP
2. **Loop unrolling** to reduce branch overhead
3. **Optimized threading** with proper work distribution
4. **Prefetching** to hide memory latency

## Recommendations

### Why 2048x2048 is Slower

The root cause is **optimization heuristics** - the optimizer chooses different optimizations based on array size:
- **4096x4096**: Gets UPCAST + UNROLL + THREAD (3 opts) → faster
- **2048x2048**: Gets UNROLL + THREAD (2 opts) → slower

### Root Cause Identified! 🎯

**The 1024 threshold in heuristic.py line 110 prevents UPCAST for 2048x2048**

See detailed investigation: [UPCAST_INVESTIGATION.md](UPCAST_INVESTIGATION.md)

**Key findings**:
1. UPCAST requires: `prod(output_shape[upcastable_dims]) >= 1024`
2. **2048x2048**: Gets 1 split → (8, 2048, 256) → product < 1024 → **no UPCAST**
3. **4096x4096**: Gets 2 splits → more dimensions → product >= 1024 → **gets UPCAST**

The split logic divides by 256:
- 2048/256 = 8 (too small for second split)
- 4096/256 = 16 (triggers second split)

### Immediate Action

1. **Test lowering threshold** from 1024 to 512 or 768 in [heuristic.py:110](tinygrad/codegen/opt/heuristic.py#L110)

2. **Benchmark impact** on various sizes (1024, 2048, 3072, 4096, 8192)

3. **Verify no regressions** on other operations (matmul, conv)

### Medium-term Improvements

3. **Tune optimization heuristics** - Make sure smaller reductions also get good optimizations

4. **Profile beam search** - If beam search finds better kernels, update heuristics

5. **Fix vector_acc for all SPLIT_REDUCEOP cases** - Ensure vectorization works across all split patterns

## Testing Commands

### Standard benchmark (use this for all testing):
```bash
# With threading (default)
DEBUG=0 CPU=1 CPU_LLVM=1 BEAM=0 IGNORE_BEAM_CACHE=1 python3 test/speed/external_test_speed_v_torch.py TestSpeed.test_sum

# Without threading
THREADS=0 DEBUG=0 CPU=1 CPU_LLVM=1 BEAM=0 IGNORE_BEAM_CACHE=1 python3 test/speed/external_test_speed_v_torch.py TestSpeed.test_sum

# With beam search (to find better optimizations)
BEAM=2 DEBUG=0 CPU=1 CPU_LLVM=1 IGNORE_BEAM_CACHE=1 python3 test/speed/external_test_speed_v_torch.py TestSpeed.test_sum
```

### Debug kernel generation:
```bash
# See what optimizations are applied
DEBUG=3 CPU=1 CPU_LLVM=1 BEAM=0 python -c "
from tinygrad import Tensor; import numpy as np
a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()
a.sum().realize()
" 2>&1 | grep -E "(split|Opt\()"

# See generated C code
DEBUG=5 CPU=1 CPU_LLVM=1 BEAM=0 python -c "
from tinygrad import Tensor; import numpy as np
a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()
a.sum().realize()
" 2>&1 | grep -A 30 "void r_"
```

## Key Files

- [tinygrad/renderer/llvmir.py:136-149](tinygrad/renderer/llvmir.py#L136-L149) - Threading implementation
- [tinygrad/codegen/late/devectorizer.py:225-324](tinygrad/codegen/late/devectorizer.py#L225-L324) - Vectorization and vector_acc logic
- [baseline_reduce_benchmark.py](baseline_reduce_benchmark.py) - Benchmark with SPLIT_REDUCEOP=0
