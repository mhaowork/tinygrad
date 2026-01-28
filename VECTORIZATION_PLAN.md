# Plan: Fix Tinygrad's Reduce Vectorization

## Current Problem

Tinygrad generates code that:
1. ✅ Loads vectors from memory (`ldr q1`)
2. ❌ Immediately extracts to scalars (`mov s2, v1.s[1]`)
3. ❌ Does scalar arithmetic (`fadd s0, s2, s0`)

Result: **4 GB/s** (1/4 of potential speed)

## Target Goal

Generate code that:
1. ✅ Loads vectors
2. ✅ Keeps data in vector registers
3. ✅ Does vector arithmetic (`fadd v1.4s, v1.4s, v2.4s`)

Result: **~60 GB/s** (matching PyTorch)

---

## Implementation Plan

### Phase 1: Understand Current Code Generation

**Files to investigate:**
1. `tinygrad/codegen/` - Code generation logic
2. `tinygrad/renderer/cstyle.py` - C code renderer
3. `tinygrad/uop/ops.py` - UOp definitions

**Key questions:**
- Where does tinygrad decide to use scalar vs vector operations?
- Why does `LOAD` of a vector result in scalar extraction?
- How are `REDUCE` operations handled?

**Action items:**
```bash
# Find where reduce operations are rendered
grep -r "REDUCE" tinygrad/codegen/ tinygrad/renderer/

# Find where vector types are handled
grep -r "float4\|vector_size" tinygrad/renderer/

# Check how GEP (Get Element Pointer) extracts elements
grep -r "GEP" tinygrad/renderer/
```

---

### Phase 2: Identify the Root Cause

The UOp AST shows:
```
15 Ops.CAST     : dtypes.float.vec(4).ptr(16777216)    [14]
16 Ops.LOAD     : dtypes.float.vec(4)                  [15]
17 Ops.GEP      : dtypes.float                         [16]  (0,)
18 Ops.GEP      : dtypes.float                         [16]  (1,)
19 Ops.GEP      : dtypes.float                         [16]  (2,)
20 Ops.GEP      : dtypes.float                         [16]  (3,)
```

**Problem:** After loading `float.vec(4)`, it immediately uses `GEP` to extract scalars!

**Root cause location:**
- The renderer or optimizer is inserting GEP operations
- These get rendered as scalar extractions in C code

**Action items:**
```bash
# Find where GEP is inserted for vector loads
grep -r "Ops.GEP" tinygrad/codegen/

# Check the renderer's handling of vector operations
cat tinygrad/renderer/cstyle.py | grep -A 10 "def render"
```

---

### Phase 3: Fix the Vector Code Generation

**Option A: Prevent GEP Insertion (Ideal)**
- Modify code generation to keep vector types through the entire reduction
- Only extract to scalars at the very end (horizontal sum)

**Option B: Detect and Optimize GEP Pattern (Fallback)**
- Detect pattern: `LOAD vec(4) → GEP → ADD`
- Transform to: `LOAD vec(4) → vector ADD`

**Option C: Use Different LLVM IR (Advanced)**
- Generate LLVM IR directly instead of C
- Use LLVM vector intrinsics explicitly

**Recommended: Start with Option A**

---

### Phase 4: Implementation Steps

#### Step 1: Find the GEP insertion point
```python
# Location: tinygrad/codegen/linearizer.py or similar
# Look for code that transforms vector loads into scalar operations
```

#### Step 2: Add vector accumulator support
```python
# When doing REDUCE with ADD on vec(4):
# - Keep accumulator as vec(4)
# - Generate vector ADD operations
# - Only at the end, do horizontal reduction
```

#### Step 3: Modify C renderer
```python
# In tinygrad/renderer/cstyle.py
# When rendering ADD of vec(4) types:
# - Use vector syntax: acc += val
# - NOT: acc += val[0]; acc += val[1]; ...
```

#### Step 4: Add horizontal reduction at end
```python
# After loop:
# result = acc[0] + acc[1] + acc[2] + acc[3]
# Or use NEON intrinsics: vaddvq_f32(acc)
```

---

### Phase 5: Testing Strategy

#### Test 1: Verify C code generation
```python
# After changes, check generated C code
# Should see:
#   float4 acc = {0,0,0,0};
#   acc += val;  // vector add
# NOT:
#   float acc = 0;
#   acc += val[0];  // scalar add
```

#### Test 2: Verify assembly
```bash
# Check for vector instructions
# Should see: fadd v0.4s, v0.4s, v1.4s
# NOT: fadd s0, s0, s1
```

#### Test 3: Benchmark performance
```bash
# Should achieve ~60 GB/s (warm)
# Currently: 4 GB/s
```

---

## Alternative: Quick Fix via Optimization Pass

If modifying core code generation is too complex, add a post-processing optimization:

**Location:** `tinygrad/codegen/opt/` or similar

**Pattern to detect:**
```python
# Detect:
for i in range(N):
    v = load_vec4(data[i*4])
    acc += v[0]
    acc += v[1]
    acc += v[2]
    acc += v[3]

# Transform to:
vec4 acc_v = {0,0,0,0}
for i in range(N):
    acc_v += load_vec4(data[i*4])
acc = acc_v[0] + acc_v[1] + acc_v[2] + acc_v[3]
```

---

## Expected Outcome

**Before:**
- 4 GB/s (scalar operations)
- Assembly: `fadd s0, s0, s1`

**After:**
- 60 GB/s (vector operations)
- Assembly: `fadd v0.4s, v0.4s, v1.4s`

**Performance gain: 15×**

---

## Next Steps

1. **Investigate** code generation (Phase 1)
2. **Locate** where GEP is inserted (Phase 2)
3. **Implement** fix (Phase 3-4)
4. **Test** and verify (Phase 5)

Would you like me to start with Phase 1 investigation?
