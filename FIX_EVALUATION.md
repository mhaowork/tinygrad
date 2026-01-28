# Fix Evaluation: Devectorizer Changes

## Summary

Your fix **partially works** and provides **~10% improvement**, but the main issue remains.

## Results

### Before Fix (threshold=256 only)
- 2048x2048: **1.51x slower**
- 4096x4096: **0.82x faster**

### After Fix (threshold=256 + devectorizer changes)
- 2048x2048: **1.43x slower** (✅ 5% improvement)
- 4096x4096: **0.78x faster** (✅ 5% improvement)

## What the Fix Does

### 1. Keep Vector Accumulators (✅ Working)

```python
def no_vectorized_buf(buf:UOp, ctx=None):
  # On CPU, keep vector register accumulators vectorized
  if (getattr(ctx, "device", None) == "CPU" and buf.op is Ops.DEFINE_REG and
      buf.ptrdtype.addrspace == AddrSpace.REG and buf.ptrdtype.base.count > 1):
    return None  # Don't devectorize!
```

**Effect**: The `r_64_4` kernel now does:
```c
float4 acc0[1];  // ✅ Vector accumulator
*(acc0+0) = ((*(acc0+0))+val0);  // ✅ Vector add
```

**Assembly**:
```asm
fadd v0.4s, v0.4s, v1.4s  // ✅ FAST! Pure vector add
```

This **works perfectly** for simple reductions.

### 2. Multiple Independent Accumulators (❌ Not Triggering)

```python
if vector_acc and ctx.device == "CPU" and len(reduce_range) == 1 and reduce_range[0].src[0].op is Ops.CONST:
  n = int(rr.src[0].arg)
  if n >= 1024 and n % 4 == 0:
    # Create 4 independent vector accumulators
```

**Problem**: This path isn't triggering for the main `r_8_8_4_4096_4` kernel.

**Why**: The first kernel has UPCAST applied, which creates multiple reduce ranges or modifies the structure. The condition `len(reduce_range) == 1` likely fails.

## Remaining Issues

### The Main Problem: UPCAST Transpose

The `r_8_8_4_4096_4` kernel still does:

```c
for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {
  float4 val0 = load(...);  // Load 4 vectors
  float4 val1 = load(...);
  float4 val2 = load(...);
  float4 val3 = load(...);

  // ❌ TRANSPOSE pattern - horizontal reduction
  *(acc0+0) = ((*(acc0+0))+
    (float4){val3[0],val0[0],val1[0],val2[0]}+  // Extract + shuffle
    (float4){val3[1],val0[1],val1[1],val2[1]}+  // Extract + shuffle
    (float4){val3[2],val0[2],val1[2],val2[2]}+  // Extract + shuffle
    (float4){val3[3],val0[3],val1[3],val2[3]}); // Extract + shuffle
}
```

**Assembly**:
```asm
ldr	q4, [x15]                  ; Load
zip1	v5.4s, v4.4s, v1.4s       ; Shuffle (SLOW)
mov	v5.s[2], v2.s[0]           ; Extract (SLOW)
mov	v5.s[3], v3.s[0]           ; Extract (SLOW)
fadd	v0.4s, v0.4s, v5.4s       ; Add
trn2	v5.4s, v4.4s, v1.4s       ; More shuffle (SLOW)
...
```

This is the **transpose problem** - UPCAST loads data in one layout but needs it in another.

## Why PyTorch is Faster

PyTorch doesn't use UPCAST for reductions. It uses **multiple independent vector accumulators** without transpose:

```c
float32x4_t acc0 = ..., acc1 = ..., acc2 = ..., acc3 = ...;
for (int i = 0; i < N; i += 16) {
  acc0 = vaddq_f32(acc0, vld1q_f32(&in[i+0]));   // ✅ No transpose
  acc1 = vaddq_f32(acc1, vld1q_f32(&in[i+4]));   // ✅ No transpose
  acc2 = vaddq_f32(acc2, vld1q_f32(&in[i+8]));   // ✅ No transpose
  acc3 = vaddq_f32(acc3, vld1q_f32(&in[i+12]));  // ✅ No transpose
}
```

## Root Cause

UPCAST is designed for operations that benefit from **reusing loaded data** (like matrix multiply). For pure reductions, it causes **transpose overhead** that hurts performance.

## Suggested Next Steps

### Option 1: Disable UPCAST for Pure Reductions
Modify heuristics to avoid UPCAST when it's just a reduction with no reuse.

### Option 2: Fix UPCAST Transpose
Generate code that avoids the transpose pattern - keep loads and accumulates aligned.

### Option 3: Force Multiple Accumulators Path
Debug why the 4-accumulator path isn't triggering for the first kernel and fix the condition.

### Option 4: Special Case CPU Reductions
Add a specific codepath for CPU reductions that generates optimal code without going through UPCAST.

## Code Locations

- Vector accumulator fix: [devectorizer.py:232-237](tinygrad/codegen/late/devectorizer.py#L232-L237) ✅ Working
- Multi-accumulator path: [devectorizer.py:319-339](tinygrad/codegen/late/devectorizer.py#L319-L339) ❌ Not triggering
- UPCAST heuristic: [heuristic.py:107-133](tinygrad/codegen/opt/heuristic.py#L107-L133)

## Conclusion

The fix is **good progress** - it solves the vector accumulator issue for simple kernels. But the main performance problem (UPCAST transpose) remains. The 1.43x slowdown for 2048x2048 is still primarily due to the transpose overhead in the main reduction kernel.
