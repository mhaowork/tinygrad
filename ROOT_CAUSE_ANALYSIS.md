# Root Cause Analysis: Why Tinygrad Doesn't Use Vector Operations

## Summary

Tinygrad's reduce operations are **15× slower** than they should be because `horizontal_reduce()` in `devectorizer.py` extracts vector elements to scalars **inside the reduction loop** instead of keeping them as vectors.

## The Problem

**Current behavior:**
```c
float acc = 0.0f;
for (int i = 0; i < N; i++) {
    float4 val = load_vec4(data[i]);
    acc += val[0];  // ← scalar add
    acc += val[1];  // ← scalar add
    acc += val[2];  // ← scalar add
    acc += val[3];  // ← scalar add
}
```

**Desired behavior:**
```c
float4 acc = {0,0,0,0};
for (int i = 0; i < N; i++) {
    float4 val = load_vec4(data[i]);
    acc += val;  // ← vector add (4 operations at once!)
}
float result = acc[0] + acc[1] + acc[2] + acc[3];
```

## Root Cause Location

**File:** `tinygrad/codegen/late/devectorizer.py`

### Function 1: `horizontal_reduce()` (lines 289-295)

```python
def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount)))
            for i in range(0, horizontal_amount)]
  return [inp]
```

**What it does:**
- Input: `float.vec(4)` vector
- Output dtype: `float` (scalar)
- Since they differ, extracts 4 scalars: `[inp.gep((0,)), inp.gep((1,)), inp.gep((2,)), inp.gep((3,))]`

### Function 2: `reduce_to_acc()` (lines 297-314)

```python
def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]
  lst = horizontal_reduce(inp, red.dtype)  # ← PROBLEM: extracts to scalars!
  # ...
  ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)  # ← adds scalars
```

**What it does:**
- Calls `horizontal_reduce(inp, red.dtype)` where `red.dtype` is the final scalar type
- This extracts the vector to scalars **before** the accumulation loop
- Then adds these scalars one by one using `functools.reduce`

## The Call Stack

```
a.sum()
  ↓
REDUCE UOp created with dtype=float (scalar result)
  ↓
reduce_to_acc(red) called
  ↓
lst = horizontal_reduce(inp, red.dtype)
  ↓
inp.dtype = float.vec(4), red.dtype = float
  ↓
Creates: [inp.gep((0,)), inp.gep((1,)), inp.gep((2,)), inp.gep((3,))]
  ↓
functools.reduce adds them: acc + val[0] + val[1] + val[2] + val[3]
  ↓
C renderer outputs: acc += val[0]; acc += val[1]; ...
  ↓
LLVM generates scalar assembly: fadd s0, s0, s1
```

## Why DEVECTORIZE=0 Doesn't Help

Setting `DEVECTORIZE=0` only disables the `devectorize` pattern matcher (line 256-262), which is a **separate** optimization that runs later.

The `horizontal_reduce()` function is **always called** during `reduce_to_acc()`, regardless of DEVECTORIZE setting.

## Evidence

### UOp AST (with DEVECTORIZE=0):
```
16 Ops.LOAD     : dtypes.float.vec(4)     [15]         ← Vector load ✓
17 Ops.GEP      : dtypes.float            [16]  (0,)   ← Extract element 0 ✗
18 Ops.GEP      : dtypes.float            [16]  (1,)   ← Extract element 1 ✗
19 Ops.GEP      : dtypes.float            [16]  (2,)   ← Extract element 2 ✗
20 Ops.GEP      : dtypes.float            [16]  (3,)   ← Extract element 3 ✗
21 Ops.ADD      : dtypes.float            [12, 17]     ← Scalar add ✗
```

### Generated C Code:
```c
float4 val0 = (*((float4*)((data1_16777216+(Ridx0<<2)))));
*(acc0+0) = ((*(acc0+0))+val0[0]+val0[1]+val0[2]+val0[3]);
```

### Assembly:
```asm
ldr   q1, [x1, x8, lsl #4]   ; Load vector ✓
fadd  s0, s0, s1              ; Scalar add ✗
mov   s2, v1.s[1]             ; Extract element ✗
fadd  s0, s2, s0              ; Scalar add ✗
```

### Performance:
- **Current:** 4 GB/s (scalar operations)
- **Potential:** 60 GB/s (vector operations)
- **Loss:** **15× slower!**

## Why This Was Done This Way

Looking at the comment on line 290-292:
```python
# if this has a horizontal reduction component, do that first
# NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
```

The function is designed to handle cases where you want to reduce a `vec(8)` to a `vec(4)`:
- Input: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Output: `[0+4, 1+5, 2+6, 3+7]`

This is useful for reducing vector width while keeping vector operations.

**However**, when reducing to a **scalar** (`vec(4)` → `float`), it should:
1. Keep the accumulator as `vec(4)` during the loop
2. Do vector ADD operations
3. Only at the very end, do horizontal reduction to get the final scalar

## The Fix

The fix needs to modify `reduce_to_acc()` to:

1. Detect when `inp` is a vector and `red.dtype` is a scalar
2. Create a **vector accumulator** instead of scalar
3. Keep vector operations in the loop
4. Only after the loop ends, call `horizontal_reduce()` to get the final scalar

**Pseudocode:**
```python
def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]

  # NEW: Check if we should use vector accumulator
  if inp.dtype.count > 1 and red.dtype.count == 1:
    # Use vector accumulator
    acc_dtype = inp.dtype  # Keep as vector!
    acc = UOp(Ops.DEFINE_REG, acc_dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=ctx.acc_num)
    # ... initialize and do vector ADD in loop ...
    # THEN at the end, do horizontal reduction
    lst = horizontal_reduce(result, red.dtype)
    return functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
  else:
    # Original scalar path
    lst = horizontal_reduce(inp, red.dtype)
    # ...
```

## Expected Results After Fix

**C Code:**
```c
float4 acc = {0,0,0,0};
for (int i = 0; i < N; i++) {
    float4 val = load_vec4(data[i]);
    acc += val;  // Vector add
}
result = acc[0] + acc[1] + acc[2] + acc[3];
```

**Assembly:**
```asm
ldr   q1, [x1, x8, lsl #4]   ; Load vector
fadd  v0.4s, v0.4s, v1.4s    ; Vector add (4 floats at once!)
```

**Performance:**
- **Before:** 4 GB/s
- **After:** 60 GB/s
- **Speedup:** 15×

## Why LLVM Renderer Won't Help Either

Even though LLVM renderer (llvmir.py) supports vectors natively:
- Line 12: `ldt()` renders `<4 x float>` LLVM vector syntax
- Line 137: `supports_float4 = True`
- Line 105-106: Binary ops work on vector types directly

**The problem is upstream:** By the time it reaches the renderer, the UOps already have GEP operations that extract to scalars. The renderer just faithfully translates what it's given.

Setting `CPU_LLVM=1` would use LLVM IR instead of C, but would still generate scalar operations because the UOps are already scalar.

## Conclusion

The fix must be in `reduce_to_acc()` to:
1. Defer horizontal reduction until after the accumulation loop
2. Use vector accumulator when beneficial
3. Only extract to scalars at the very end

This is a strategic change to the reduction logic, not a simple renderer or devectorization tweak.
