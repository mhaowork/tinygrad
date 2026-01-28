# Tinygrad Vectorization Investigation Report

## Root Cause Found

**File:** `tinygrad/codegen/late/devectorizer.py`
**Function:** `no_vectorized_alu` (lines 225-229)

```python
def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  if alu.op is Ops.WHERE and alu.src[2].arg is Invalid: return None
  # THIS IS THE PROBLEM:
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg)
               for i in range(alu.dtype.vcount))
  return UOp(Ops.VECTORIZE, alu.dtype, alus)
```

### What This Does

For a vector ADD like `acc0 += val0` where both are `float4`:

1. **Extracts each element:** `val0.gep(0), val0.gep(1), val0.gep(2), val0.gep(3)`
2. **Creates scalar operations:** `acc[0]+val[0], acc[1]+val[1], acc[2]+val[2], acc[3]+val[3]`
3. **Re-wraps in VECTORIZE:** `{result0, result1, result2, result3}`

### Why This Is Bad

The C renderer (lines 19-21 of `cstyle.py`) renders this as:
```c
float4 result = {acc[0]+val[0], acc[1]+val[1], acc[2]+val[2], acc[3]+val[3]};
```

But LLVM/Clang sees individual scalar accesses and generates:
```asm
ldr   q1, [x1, x8, lsl #4]   ; Load vector
mov   s2, v1.s[1]             ; Extract element [1] to scalar
fadd  s0, s2, s0              ; Scalar add
mov   s2, v1.s[2]             ; Extract element [2] to scalar
fadd  s0, s2, s0              ; Scalar add
...
```

Instead of the desired:
```asm
ldr   q1, [x1, x8, lsl #4]   ; Load vector
fadd  v0.4s, v0.4s, v1.4s    ; Vector add (4 floats at once!)
```

---

## Why Was It Written This Way?

Looking at line 256-262, the `devectorize` pattern matcher is used to:
> "no ALU on vectorized dtypes"

**Purpose:** Ensure compatibility with renderers that don't support vector operations.

**Problem:** It's being applied even to `ClangRenderer` which **does** support vector operations!

---

## The Call Stack

```
a.sum()
  ↓
REDUCE operation created
  ↓
reduce_to_acc() in devectorizer.py (line 297-314)
  ↓
horizontal_reduce() extracts with GEP (line 289-295)
  ↓
functools.reduce(lambda x,y: x.alu(red.arg, y), lst)  (line 312)
  ↓
Creates ADD UOp with vector dtype
  ↓
devectorize pattern matcher runs (line 256-262)
  ↓
no_vectorized_alu() de-vectorizes the ADD (line 225-229)
  ↓
Scalar operations generated
  ↓
C renderer outputs scalar code
  ↓
LLVM generates scalar assembly
```

---

## Evidence

### UOp AST (from earlier debug output):
```
15 Ops.CAST     : dtypes.float.vec(4).ptr(16777216)    [14]
16 Ops.LOAD     : dtypes.float.vec(4)                  [15]    ← Vector load
17 Ops.GEP      : dtypes.float                         [16]  (0,) ← Extract!
18 Ops.GEP      : dtypes.float                         [16]  (1,) ← Extract!
19 Ops.GEP      : dtypes.float                         [16]  (2,) ← Extract!
20 Ops.GEP      : dtypes.float                         [16]  (3,) ← Extract!
21 Ops.ADD      : dtypes.float                         [12, 17]  ← Scalar add
```

### Generated C Code:
```c
float acc0[1];
*(acc0+0) = 0.0f;
for (int Ridx0 = 0; Ridx0 < 4194304; Ridx0++) {
    float4 val0 = (*((float4*)((data1_16777216+(Ridx0<<2)))));
    *(acc0+0) = ((*(acc0+0))+val0[0]+val0[1]+val0[2]+val0[3]);  // ← Scalar ops!
}
```

### Assembly Output:
```asm
ldr   q1, [x1, x8, lsl #4]   ; Load vector
fadd  s0, s0, s1              ; Scalar add ← BAD!
mov   s2, v1.s[1]
fadd  s0, s2, s0              ; Scalar add ← BAD!
```

### Performance Impact:
- **Current**: 4 GB/s (scalar operations)
- **Potential**: 60 GB/s (vector operations)
- **Loss**: **15× slower!**

---

## Solution Options

### Option 1: Conditional Devectorization (Recommended)

Modify `devectorize` pattern matcher to skip for renderers that support vectors:

**Location:** `tinygrad/codegen/late/devectorizer.py` line 256

**Change:**
```python
devectorize = PatternMatcher([
  # CAST after AFTER
  (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"),
   lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
  # no ALU on vectorized dtypes - BUT ONLY FOR RENDERERS WITHOUT VECTOR SUPPORT
  # Add check: if renderer.supports_vectors, skip devectorization
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
])+devectorize_buf_and_index
```

### Option 2: Add Vector-Aware Reduction

Modify `reduce_to_acc` to keep accumulator as vector type:

**Location:** `tinygrad/codegen/late/devectorizer.py` line 297-314

**Change:**
```python
def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]

  # NEW: Don't do horizontal reduction if input is vector
  if inp.dtype.count > 1 and red.dtype.count > 1:
    # Keep as vector accumulator
    acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=ctx.acc_num)
    # Vector ADD directly
    lst = [acc.after(...).index(...)] + [inp]
    ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
    # Only do horizontal reduction at the very end
  else:
    # Original scalar path
    lst = horizontal_reduce(inp, red.dtype)
    ...
```

### Option 3: Add Renderer Capability Flag

Add flag to renderers indicating vector support:

**Location:** `tinygrad/renderer/cstyle.py`

```python
class ClangRenderer(CStyleLanguage):
  device = "CPU"
  supports_vector_alu = True  # NEW FLAG
  ...
```

Then check this flag in devectorizer before applying `no_vectorized_alu`.

---

## Recommended Fix

**Start with Option 1** (conditional devectorization) because:
1. Minimal code changes
2. Affects only the devectorizer
3. Preserves compatibility with non-vector renderers
4. Should immediately improve performance for Clang/Metal/CUDA

**Implementation Steps:**

1. Add `supports_vector_alu` flag to renderer classes
2. Pass renderer context to devectorizer
3. Check flag in `no_vectorized_alu`:
   ```python
   def no_vectorized_alu(ctx, alu:UOp):
     if ctx and ctx.supports_vector_alu and alu.op in {Ops.ADD, Ops.MUL}:
       return None  # Keep vector operations
     # Otherwise, devectorize as before
   ```

4. Test and verify assembly improves

---

## Expected Results

**Before Fix:**
```asm
fadd  s0, s0, s1              # 1 float per instruction
mov   s2, v1.s[1]
fadd  s0, s2, s0
```
Performance: 4 GB/s

**After Fix:**
```asm
fadd  v0.4s, v0.4s, v1.4s     # 4 floats per instruction
```
Performance: 60 GB/s

**Speedup: 15×**

---

## Next Steps

Would you like me to:
1. Implement Option 1 (conditional devectorization)?
2. Create a test case to verify the fix?
3. Check if there are other operations being incorrectly devectorized?
