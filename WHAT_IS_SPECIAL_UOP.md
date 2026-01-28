# What is a SPECIAL UOp?

## TL;DR

`SPECIAL` is a UOp that represents **hardware-specific runtime indices** like thread IDs or workgroup IDs. It's tinygrad's way of saying: "This is a value that comes from the hardware at runtime, not from computation."

## Definition

**File: [tinygrad/uop/__init__.py:20](tinygrad/uop/__init__.py#L20)**
```python
# this is a RANGE for GPU dimensions, similar to symbolic shapes but not exactly
SPECIAL = auto()
```

**Key properties:**
- Represents runtime indices provided by hardware (thread ID, workgroup ID, etc.)
- **NOT computed** - value comes from outside the kernel
- Analogous to CUDA's `threadIdx.x`, `blockIdx.x`, OpenCL's `get_global_id()`, etc.
- Tagged as `Irreducible` - cannot be simplified or rewritten

## How It's Created

**File: [tinygrad/codegen/gpudims.py:38](tinygrad/codegen/gpudims.py#L38)**

```python
def get_grouped_dims(prefix, dims, max_sizes, reverse=False):
    # ...
    # Creates SPECIAL UOps for each dimension
    raw_idxs = [UOp(Ops.SPECIAL, dtypes.index, (sint_to_uop(s),), (f"{prefix}{i}"))
                for i,s in enumerate(limited)]
    # → [SPECIAL("gidx0"), SPECIAL("gidx1"), SPECIAL("gidx2")]
    return raw_idxs
```

**Created during codegen** when `add_gpudims()` runs:
1. Finds all THREAD/GLOBAL axes in the UOp graph
2. Creates SPECIAL UOps to represent thread indices
3. Substitutes these into the graph to replace RANGE(THREAD) UOps

## Structure of a SPECIAL UOp

```python
UOp(Ops.SPECIAL, dtype=dtypes.index, src=(UOp.const(8),), arg="gidx0")
#   ↑            ↑                    ↑                   ↑
#   |            |                    |                   |
#   Op type      Data type            Max value           Name/identifier
```

**Components:**
- **dtype**: Always `dtypes.index` (integer type for indexing)
- **src[0]**: Maximum value (e.g., 8 means 0..7 range)
- **arg**: String identifier (e.g., "gidx0", "lidx1", "idx0")

**Common names:**
- `gidx{N}` - Global thread index (dimension N)
- `lidx{N}` - Local workgroup index (dimension N)
- `idx{N}` - Generic index (for CPU without local/global distinction)

## What SPECIAL Represents

**Hardware-specific indices:**

| Name | CPU | GPU (CUDA/OpenCL) | Meaning |
|------|-----|-------------------|---------|
| `gidx0` | `core_id` | `blockIdx.x * blockDim.x + threadIdx.x` | Global thread ID (x) |
| `gidx1` | - | `blockIdx.y * blockDim.y + threadIdx.y` | Global thread ID (y) |
| `gidx2` | - | `blockIdx.z * blockDim.z + threadIdx.z` | Global thread ID (z) |
| `lidx0` | - | `threadIdx.x` | Local thread ID (x) |
| `lidx1` | - | `threadIdx.y` | Local thread ID (y) |
| `idx0` | `core_id` | Same as `gidx0` | Generic thread ID |

## How It's Used

### Step 1: Graph Contains RANGE(THREAD)
```python
RANGE(0..8, THREAD)  # Thread axis with 8 threads
```

### Step 2: Codegen Replaces with SPECIAL
```python
# In add_gpudims():
subs[RANGE(0..8, THREAD)] = SPECIAL("gidx0", max=8)
graph.substitute(subs)
```

### Step 3: All Index Operations Updated
```python
# Before:
INDEX = (RANGE(THREAD) * stride) + offset

# After substitution:
INDEX = (SPECIAL("gidx0") * stride) + offset
```

### Step 4: Renderer Converts to Target Language

**ClangJIT (CPU):**
```python
# Pattern in ClangRenderer
code_for_workitem = {"g": lambda _: "core_id"}

# SPECIAL("gidx0") becomes:
int gidx0 = core_id;
```

**LLVM (after PR #13781):**
```python
# Pattern in LLVMRenderer
(UPat(Ops.SPECIAL, name="x"),
 lambda ctx,x: f"{ctx[x]} = add i32 %core_id, 0")

# SPECIAL("gidx0") becomes:
%gidx0 = add i32 %core_id, 0
```

**CUDA:**
```python
# SPECIAL("gidx0") becomes:
int gidx0 = blockIdx.x * blockDim.x + threadIdx.x;
```

**OpenCL:**
```python
# SPECIAL("gidx0") becomes:
int gidx0 = get_global_id(0);
```

## Example: Full Flow

### Python Code
```python
t = Tensor.rand(1024)
result = t.sum().realize()
```

### UOp Graph (after threading optimization)
```
SPECIAL("gidx0", max=8)  ← Created by add_gpudims
    ↓
MUL(SPECIAL("gidx0"), 128)  ← Each thread handles 128 elements
    ↓
ADD(..., RANGE(0..128, LOOP))  ← Plus loop within thread
    ↓
INDEX(data, ...)  ← Memory access using gidx0
```

### Generated Code
```c
void kernel(float* data, int core_id) {
    int gidx0 = core_id;  // ← SPECIAL rendered

    for (int i = 0; i < 128; i++) {
        int idx = (gidx0 * 128) + i;  // ← Uses gidx0 for partitioning
        sum += data[idx];
    }
}
```

### Runtime Execution
```
Thread 0: core_id=0 → gidx0=0 → processes data[0..127]
Thread 1: core_id=1 → gidx0=1 → processes data[128..255]
...
Thread 7: core_id=7 → gidx0=7 → processes data[896..1023]
```

## Why SPECIAL Instead of Just Variables?

### 1. **Hardware Abstraction**
SPECIAL represents values that **come from hardware**, not computation:
- GPU: From thread/block IDs
- CPU: From core_id parameter
- Different for each parallel execution instance

### 2. **Cannot Be Simplified**
```python
# Regular UOp:
ADD(CONST(2), CONST(3))  → Can simplify to CONST(5)

# SPECIAL UOp:
SPECIAL("gidx0")  → Cannot simplify! Value unknown until runtime
```

Marked as `Irreducible` in [GroupOp](tinygrad/uop/__init__.py#L115):
```python
Irreducible = {Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE}
```

### 3. **Renderer-Specific**
Each target has different ways to access thread IDs:
- SPECIAL provides a **unified representation**
- Renderer converts to appropriate hardware intrinsic
- Same UOp graph works on all backends!

### 4. **Bounds Known at Compile Time**
```python
SPECIAL("gidx0", max=8)
```
Even though the **value** is runtime, the **range** is known:
- Optimizer knows: `0 <= gidx0 < 8`
- Can use for bounds checking, optimization
- Stored in `src[0]` of the SPECIAL UOp

## SPECIAL vs RANGE vs CONST

| UOp Type | Value Known | Source | Example |
|----------|-------------|--------|---------|
| `CONST` | Compile-time | Literal | `CONST(42)` → `42` |
| `RANGE` | Iteration | Loop variable | `RANGE(0..10, LOOP)` → `for i=0..9` |
| `SPECIAL` | **Runtime** | **Hardware** | `SPECIAL("gidx0")` → `core_id` |

**Key difference:**
- CONST: Value is in the code
- RANGE: Value from loop iteration
- SPECIAL: Value from hardware/runtime environment

## Practical Example: Comparing All Three

```python
# Original operation: data[i] = i * 2 + 10

# With CONST:
STORE(data, CONST(10))  # Always stores 10

# With RANGE:
i = RANGE(0..100, LOOP)
STORE(data[i], i * 2 + 10)  # Stores 10, 12, 14, ..., 208

# With SPECIAL:
gid = SPECIAL("gidx0", max=10)  # 10 threads
i = RANGE(0..10, LOOP)
idx = gid * 10 + i
STORE(data[idx], idx * 2 + 10)
# Thread 0 stores: data[0..9] = 10, 12, ..., 28
# Thread 1 stores: data[10..19] = 30, 32, ..., 48
# ...
```

## In the Context of Threading

**SPECIAL is the bridge between:**
1. **Abstract parallelism** (UOp graph says "parallelize this")
2. **Concrete hardware** (actual thread ID from CPU/GPU)

**Flow:**
```
Optimizer: "Split into 8 threads" → RANGE(0..8, THREAD)
                ↓
Codegen: "Replace with thread index" → SPECIAL("gidx0", max=8)
                ↓
Renderer: "Get from hardware" → int gidx0 = core_id;
                ↓
Runtime: Executes with core_id=0,1,2,...,7
```

## Summary

**What is SPECIAL?**
- A UOp representing hardware-provided runtime indices
- Like thread ID, workgroup ID, core ID, etc.

**Why does it exist?**
- Abstract away hardware-specific thread ID mechanisms
- Allow same UOp graph to work on CPU, CUDA, OpenCL, Metal, etc.
- Preserve range information for optimization

**How is it used?**
- Created during codegen to replace THREAD/GLOBAL ranges
- Used in index calculations for memory access
- Rendered to target-specific code by each backend

**Key insight:**
SPECIAL is tinygrad's way of saying: "This is a value I need from the hardware at runtime to partition work across parallel execution units."
