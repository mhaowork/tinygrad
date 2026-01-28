## Vector accumulator change (what it did and why)

This note explains the two tweaks I made when trying to get `a.sum()` to emit the same style of vector C as `extra/reduce_speed_simple.py`. Both changes live in `tinygrad/codegen/late/devectorizer.py`.

### 1) Allow vector ADD to survive devectorize on CPU

- **Before:** `no_vectorized_alu` always split vector ALU into per-lane scalars. Even if we had a `float4` add, it became four scalar adds.
- **After (conditional):** If the device is CPU and the op is `ADD` on a float vector, we *keep* the vector form. That lets a `float4` accumulator stay a vector across the loop so the renderer can emit `acc0 += val0;` instead of lane-by-lane adds.

### 2) Keep the reduction accumulator vectorized until the end

- **Before:** `reduce_to_acc` calls `horizontal_reduce` immediately when input is wider than output. For a sum returning scalar, a `float4` load is split into four scalars and reduced on the spot. The accumulator is scalar, so the generated C looks like:
  ```c
  float acc0 = 0;
  for (...) {
    float4 val0 = *(float4*)(...);
    acc0 = acc0 + val0[0] + val0[1] + val0[2] + val0[3];
  }
  ```
  This is a long dependency chain of scalar adds.
- **After:** If the reduce has a range (loop) and the input is vector but the output is scalar, we create a *vector* accumulator. We accumulate with vector `ADD` in the loop, and only perform one horizontal reduce at the end. The emitted C becomes:
  ```c
  float4 acc0 = {0,0,0,0};
  for (...) {
    float4 val0 = *(float4*)(...);
    acc0 = acc0 + val0;        // vector add
  }
  float acc_scalar = acc0[0] + acc0[1] + acc0[2] + acc0[3];
  ```
  This mirrors `extra/reduce_speed_simple.py` and removes the scalar dependency chain inside the loop.

### Concrete toy example

Consider summing a `(2, 4)` float tensor on CPU with `SPLIT_REDUCEOP=0`:

- **Original lowering:**  
  - Load: one `float4` per row.  
  - `horizontal_reduce` immediately splits to four scalars.  
  - Loop body does `acc0 = acc0 + v0 + v1 + v2 + v3;` (scalar accumulator).  
  - Renderer outputs a scalar add chain per iteration.

- **With the vector-accumulator tweak:**  
  - Load: `float4` per row.  
  - Accumulator: `float4 acc0`.  
  - Loop body does `acc0 += val0;` (vector add).  
  - After the loop, one horizontal reduce: `acc_scalar = acc0[0] + acc0[1] + acc0[2] + acc0[3];`.  
  - Renderer keeps the vector add because of the devectorize relaxation above.

### ASCII UOps graphs (before vs after)

Example: `a.sum(axis=1)` on shape `(2, 3)`:

**Before `reduce_to_acc` (abstract REDUCE)**
```
DEFINE_GLOBAL data0
DEFINE_GLOBAL data1
RANGE r0
RANGE r1
  INDEX data1[r0, r1] -> LOAD val
  REDUCE(val, r1, arg=ADD) -> red
  STORE data0[r0] = red
```

**After `reduce_to_acc` (explicit accumulator)**
```
DEFINE_GLOBAL data0
DEFINE_GLOBAL data1
DEFINE_REG acc0 (size=1)

acc0.after(r0).index(0).store(0)          # init per outer loop

RANGE r0
  RANGE r1
    acc0.after(acc_init, r1).index(0) -> acc_load
    INDEX data1[r0, r1] -> LOAD val
    ADD acc_load, val -> ret
    acc0.index(0).store(ret)
  END r1
  acc0.after(store, END r1).index(0) -> acc_final
  STORE data0[r0] = acc_final
```

### Why it matters

For large reductions on CPU, the scalar add chain is latency-bound. Using a vector accumulator lets LLVM emit SIMD adds inside the loop (matching the manual kernel), improving throughput. The final horizontal reduce happens once, so the long dependency chain moves *after* the loop.

### If you want to keep `reduce_to_acc` unchanged

The scalarization is triggered inside `reduce_to_acc` via `horizontal_reduce`. If we don’t modify that function, we’d need a pre-`pm_reduce` rewrite that changes the REDUCE to produce a vector output plus a separate final horizontal fold. That’s a more invasive graph change; the above adjustments were the minimal edits to preserve vector structure during lowering.

### Patch diff with line-by-line explanations

#### `tinygrad/codegen/late/devectorizer.py` — keep vector ADDs on CPU

- `def no_vectorized_alu(alu:UOp, ctx=None):`  
  Added `ctx` so we can inspect the device during devectorize.
- `if getattr(ctx, "device", None) == "CPU" and dtypes.is_float(alu.dtype) and alu.op is Ops.ADD: return None`  
  On CPU float vectors, don’t split an `ADD` into scalars—preserve vector adds so accumulators can stay vectorized.

#### `tinygrad/codegen/late/devectorizer.py` — defer horizontal reduction

- `vector_acc = len(reduce_range) != 0 and red.arg is Ops.ADD and red.dtype.count == 1 and inp.dtype.count > 1`  
  Detect the “vector input, scalar output, has reduce loop” case, but only for `ADD` since other ops may not be safe.
- `if vector_acc: acc_dtype = red.dtype.vec(inp.dtype.count); if inp.dtype != acc_dtype: inp = inp.cast(acc_dtype)`  
  Create a vector accumulator dtype (e.g., float4) and cast the input to match it.
- `identity_dtype = acc_dtype if vector_acc else red.dtype`  
  Initialize the accumulator with the vector identity (e.g., `{0,0,0,0}`) so it matches the vector dtype.
- `acc = DEFINE_REG(ptr to acc_dtype, size=1)`  
  One register slot whose element is the vector accumulator.
- `acc_init = acc.after(*input_ranges).index(0).store(identity) if len(input_ranges) else acc.index(0).store(identity)`  
  If there are non-reduce ranges (outer loops), attach the accumulator init to those loops with `AFTER` so it runs once per outer-iteration; otherwise just store identity directly.
- `acc_load = acc.after(...).index(0)`  
  Pointer to the vector accumulator inside the loop.
- `ret = acc_load.alu(red.arg, inp)`  
  Vector `ADD`: accumulates the new vector into the vector accumulator.
- `acc_after = acc.after(acc.index(0).store(ret).end(*reduce_range)).index(0)`  
  Store the vector accumulator at loop end and reindex to fetch the post-loop vector value.
- `lst = horizontal_reduce(acc_after, red.dtype)`  
  After the loop, split the vector into scalar lanes for the final fold.
- `return functools.reduce(lambda x,y: x.alu(red.arg, y), lst)`  
  Fold the scalar lanes once to produce the final scalar output (e.g., `sum(acc0[0..3])`).
