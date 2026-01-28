# Multi-Accumulator Debug Report

## Root Cause Identified ✅

The multi-accumulator optimization path **cannot trigger** for UPCAST kernels due to a fundamental logic conflict.

## Debug Output Analysis

### First Kernel: `r_8_8_4_4096_4` (Main Reduction)
```
reduce_to_acc: vector_acc=False, len(reduce_range)=1, red.dtype=dtypes.float.vec(4), inp.dtype=dtypes.float.vec(16), device=CPU
```

**Why vector_acc is False**:
- Condition: `red.dtype.count == 1` (expects scalar output)
- Reality: `red.dtype=dtypes.float.vec(4)` → `red.dtype.count == 4`
- **UPCAST creates vectorized output**, violating the scalar output requirement

**Characteristics**:
- This kernel has **4096 iterations** (would benefit from multi-accumulator)
- But it **never gets vector_acc=True** because of vectorized output
- Multi-accumulator path never checked

### Second Kernel: `r_64_4` (Final Reduction)
```
reduce_to_acc: vector_acc=True, len(reduce_range)=1, red.dtype=dtypes.float, inp.dtype=dtypes.float.vec(4), device=CPU
vector_acc check: len(reduce_range)=1, reduce_range=(UOp(Ops.RANGE, ..., arg=64, ...),), const=[True]
multi-acc check: n=64, n>=1024=False, n%4==0=True
```

**Why multi-accumulator doesn't trigger**:
- ✅ `vector_acc=True` (scalar output)
- ✅ `len(reduce_range)==1`
- ✅ `reduce_range[0].src[0].op is Ops.CONST`
- ❌ `n=64 < 1024` (too small)

**Characteristics**:
- This kernel is **too small** (only 64 iterations)
- Correctly identified as vector reduction
- But doesn't meet size threshold for multi-accumulator

## The Fundamental Conflict

### Current Logic (devectorizer.py:306)
```python
vector_acc = len(reduce_range) != 0 and red.arg is Ops.ADD and red.dtype.count == 1 and inp.dtype.count > 1
```

This requires:
- **Scalar output** (`red.dtype.count == 1`)
- **Vector input** (`inp.dtype.count > 1`)

### What UPCAST Does
UPCAST optimization creates kernels with:
- **Vectorized output** (`red.dtype.count == 4`)
- **Vectorized input** (`inp.dtype.count == 16`)
- Large reduce range (4096)

### The Conflict
```
Multi-accumulator needs:  vector_acc=True && n >= 1024
UPCAST kernels have:      vector_acc=False (due to vectorized output)
```

**Result**: Multi-accumulator optimization **never triggers** for UPCAST kernels, even though they would benefit most!

## Why This Matters

The kernel that NEEDS multi-accumulator optimization (`r_8_8_4_4096_4` with 4096 iterations) can't get it because UPCAST vectorizes the output.

The kernel that COULD get it (`r_64_4`) is too small to benefit (only 64 iterations).

## Current Performance Impact

From [FIX_EVALUATION.md](FIX_EVALUATION.md):

**Before Fix** (threshold=256 only):
- 2048x2048: **1.51x slower**

**After Fix** (threshold=256 + devectorizer changes):
- 2048x2048: **1.43x slower** (5% improvement)

The devectorizer fix helps the small kernel (`r_64_4`) but can't help the main kernel (`r_8_8_4_4096_4`).

## The Real Problem

Looking at the generated assembly from [KERNEL_ANALYSIS.md](KERNEL_ANALYSIS.md), the main kernel still does:

```c
for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {
  float4 val3 = load(...);
  // ❌ Horizontal reduction in inner loop
  *(acc0+0) = ((*(acc0+0))+val3[0]+val3[1]+val3[2]+val3[3]);
}
```

This generates shuffle/extract instructions:
```asm
zip1  v5.4s, v4.4s, v1.4s       ; Shuffle (SLOW)
mov   v5.s[2], v2.s[0]           ; Extract (SLOW)
faddp s6, v5.2s                  ; Pairwise add (SLOW)
```

Instead of pure vector adds:
```asm
fadd v0.4s, v0.4s, v1.4s         ; FAST!
```

## Potential Solutions

### Option 1: Relax vector_acc Condition for CPU
Allow vector_acc for vectorized outputs on CPU:
```python
# Current:
vector_acc = len(reduce_range) != 0 and red.arg is Ops.ADD and red.dtype.count == 1 and inp.dtype.count > 1

# Modified:
vector_acc = (len(reduce_range) != 0 and red.arg is Ops.ADD and inp.dtype.count > 1 and
              (red.dtype.count == 1 or (ctx.device == "CPU" and red.dtype.count > 1)))
```

This would:
- Allow UPCAST kernels to use vector accumulators
- Keep vector operations throughout
- Potentially trigger multi-accumulator for large UPCAST reductions

### Option 2: Disable UPCAST for Pure Reductions on CPU
Modify heuristics to skip UPCAST for pure reduction operations on CPU, where it causes transpose overhead.

### Option 3: Fix UPCAST Code Generation
Modify UPCAST to generate code that keeps vectors aligned and avoids horizontal reduction in the inner loop.

### Option 4: Lower Multi-Accumulator Threshold
Change `n >= 1024` to something smaller like `n >= 64` so the final kernel benefits. But this doesn't solve the main kernel problem.

## Code Locations

- **vector_acc condition**: [devectorizer.py:306](tinygrad/codegen/late/devectorizer.py#L306)
- **multi-accumulator check**: [devectorizer.py:327-330](tinygrad/codegen/late/devectorizer.py#L327-L330)
- **UPCAST heuristic**: [heuristic.py:110](tinygrad/codegen/opt/heuristic.py#L110)
- **ReduceContext device**: [devectorizer.py:291-293](tinygrad/codegen/late/devectorizer.py#L291-L293)

## Next Steps

The most promising approach is **Option 1**: relax the vector_acc condition to allow vectorized outputs on CPU. This would:
1. Enable vector accumulators for UPCAST kernels
2. Potentially trigger multi-accumulator for large reductions
3. Keep the optimization working for non-UPCAST kernels

This requires careful testing to ensure:
- It generates correct code for vectorized output reductions
- The horizontal reduction is deferred to the end
- Performance improves rather than regresses
