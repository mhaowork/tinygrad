# Kernel Analysis: Tinygrad vs PyTorch (2048x2048)

## Tinygrad's Generated Kernel (First Principles)

### Main Kernel: `r_8_8_4_4096_4`

```c
void r_8_8_4_4096_4(float* restrict data0_256, float* restrict data1_4194304, int core_id) {
  float acc0[4];
  int gidx0 = core_id; /* 8 threads */

  for (int Lidx2 = 0; Lidx2 < 8; Lidx2++) {
    *(acc0+0) = 0.0f;
    *(acc0+1) = 0.0f;
    *(acc0+2) = 0.0f;
    *(acc0+3) = 0.0f;

    for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {  // INNER LOOP
      int alu4 = ((gidx0<<19)+(Lidx2<<16)+(Ridx0<<2));

      // Load 4 float4 vectors (16 floats total)
      float4 val0 = (*((float4*)((data1_4194304+(alu4+16384)))));
      float4 val1 = (*((float4*)((data1_4194304+(alu4+32768)))));
      float4 val2 = (*((float4*)((data1_4194304+(alu4+49152)))));
      float4 val3 = (*((float4*)((data1_4194304+alu4))));

      // ❌ PROBLEM: Horizontal reduction in inner loop!
      *(acc0+0) = ((*(acc0+0))+val3[0]+val3[1]+val3[2]+val3[3]);  // Unpack + add
      *(acc0+1) = ((*(acc0+1))+val0[0]+val0[1]+val0[2]+val0[3]);  // Unpack + add
      *(acc0+2) = ((*(acc0+2))+val1[0]+val1[1]+val1[2]+val1[3]);  // Unpack + add
      *(acc0+3) = ((*(acc0+3))+val2[0]+val2[1]+val2[2]+val2[3]);  // Unpack + add
    }

    *((float4*)((data0_256+((gidx0<<5)+(Lidx2<<2))))) = (float4){(*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3))};
  }
}
```

### Generated Assembly (Critical Section)

```asm
0x000028: add	x15, x13, x14              ; Address calculation
0x00002c: ldr	q1, [x15, x10]             ; Load float4 #1
0x000030: ldr	q2, [x15, x11]             ; Load float4 #2
0x000034: ldr	q3, [x15, x12]             ; Load float4 #3
0x000038: ldr	q4, [x15]                  ; Load float4 #4

; ❌ HORIZONTAL REDUCTION - extracts and shuffles
0x00003c: zip1	v5.4s, v4.4s, v1.4s       ; Shuffle/extract
0x000040: mov	v5.s[2], v2.s[0]           ; More shuffling
0x000044: mov	v5.s[3], v3.s[0]           ; More shuffling
0x000048: zip2	v1.4s, v4.4s, v1.4s       ; More shuffling
0x00004c: mov	v1.s[2], v2.s[1]           ; Extract element
0x000050: faddp	s6, v5.2s                  ; Pairwise add (horizontal)
0x000054: mov	s7, v5.s[2]                ; Extract
0x000058: fadd	s6, s7, s6                 ; Scalar add
0x00005c: mov	s5, v5.s[3]                ; Extract
0x000060: fadd	s5, s6, s5                 ; Scalar add
0x000064: fadd	s0, s0, s5                 ; Final accumulate
```

## The Fundamental Problem

### Issue 1: Horizontal Reduction in Inner Loop

Tinygrad loads `float4` vectors but immediately unpacks them:
```c
val3[0]+val3[1]+val3[2]+val3[3]  // Horizontal sum
```

This generates **shuffle + extract + pairwise add** instructions instead of vectorized adds.

### Issue 2: Scalar Accumulation

After horizontal reduction, accumulation happens in **scalar registers**:
```asm
fadd s0, s0, s5  // Scalar float add (NOT vector)
```

Instead of:
```asm
fadd v0.4s, v0.4s, v1.4s  // Vector add (what we want!)
```

## What PyTorch/Accelerate Does Differently

PyTorch on macOS uses Apple's **Accelerate framework** (vecLib), which:

1. **Keeps data in vector form** throughout the reduction
2. **Uses vector accumulators** - multiple independent `v0-v7` vector registers
3. **No horizontal reduction** in the inner loop
4. **Optimized for Apple Silicon** - takes advantage of M1/M2 NEON pipeline

### Hypothetical Optimal Code

```c
void optimal_reduce(float* out, float* in, int N) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);  // Vector accumulator #1
  float32x4_t acc1 = vdupq_n_f32(0.0f);  // Vector accumulator #2
  float32x4_t acc2 = vdupq_n_f32(0.0f);  // Vector accumulator #3
  float32x4_t acc3 = vdupq_n_f32(0.0f);  // Vector accumulator #4

  for (int i = 0; i < N; i += 16) {
    // ✅ Vector adds - NO horizontal reduction
    acc0 = vaddq_f32(acc0, vld1q_f32(&in[i + 0]));
    acc1 = vaddq_f32(acc1, vld1q_f32(&in[i + 4]));
    acc2 = vaddq_f32(acc2, vld1q_f32(&in[i + 8]));
    acc3 = vaddq_f32(acc3, vld1q_f32(&in[i + 12]));
  }

  // ✅ Horizontal reduction ONLY at the end
  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);
  *out = vaddvq_f32(acc0);  // Final horizontal sum
}
```

This generates:
```asm
ldr q0, [x1, #0]      ; Load vector
ldr q1, [x1, #16]     ; Load vector
fadd v2.4s, v2.4s, v0.4s   ; ✅ VECTOR ADD
fadd v3.4s, v3.4s, v1.4s   ; ✅ VECTOR ADD
```

## Root Cause: Devectorizer Not Fully Working

The devectorizer fix (HEAD~1) was supposed to prevent this, but it's **not applying to split reductions**.

From [devectorizer.py:301-324](tinygrad/codegen/late/devectorizer.py#L301-L324):

```python
# This fix should work for other associative Ops but let's start with ADD for now
vector_acc = len(reduce_range) != 0 and red.arg is Ops.ADD and red.dtype.count == 1 and inp.dtype.count > 1
```

The `vector_acc` path **should** create vector accumulators, but:
- It only works for simple (non-split) reductions
- With `SPLIT_REDUCEOP=1`, the reduction is split into multiple kernels
- Each split kernel doesn't trigger the `vector_acc` path

## Performance Impact

**Horizontal reduction cost** (per inner loop iteration):
- 4x `ldr` (load) - OK
- Multiple `zip1/zip2/mov` (shuffle) - **EXPENSIVE** (3-4 cycles each)
- 4x `faddp` (pairwise add) - **EXPENSIVE** (2 cycles each)
- Additional `mov` + scalar `fadd` - **EXPENSIVE**

vs **Vector addition**:
- 4x `ldr` (load) - OK
- 4x `fadd v.4s` (vector add) - **FAST** (1 cycle, pipelined)

**Estimated overhead**: ~3-4x slower per iteration!

## Why 4096 is Faster

The 4096x4096 kernel has the same problem, but:
1. Better threading efficiency (16 chunks vs 8 chunks)
2. More total work amortizes overhead
3. Better cache behavior with larger chunks

## Solution

Need to fix the devectorizer to:
1. **Recognize split reductions** and still use vector accumulators
2. **Defer horizontal reduction** until the very end
3. **Keep intermediate results in vector form**

The fix in HEAD~1 was on the right track but doesn't cover split reductions.

## Testing

```bash
# See the horizontal reduction problem
DEBUG=7 CPU=1 CPU_CLANG=1 python -c "
from tinygrad import Tensor; import numpy as np
a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()
a.sum().realize()
" 2>&1 | grep -A 50 "typedef" | grep -E "(val3|acc0)"
```

You should see: `val3[0]+val3[1]+val3[2]+val3[3]` - the horizontal reduction.
