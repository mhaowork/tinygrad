# How Threading Automatically Partitions Data

## TL;DR

The kernel is **NOT handwritten**! The partitioning happens automatically through **UOp graph transformations**:

1. Original: `RANGE(0..1024, LOOP)` → used in memory indexing
2. Optimizer: Changes to `RANGE(0..8, THREAD)` when threading applied
3. Codegen: Replaces with `SPECIAL(gidx0)`
4. All memory indices that used the loop variable now use `gidx0`
5. Since `gidx0 = core_id`, each thread automatically gets different data!

## The Complete Flow

### Step 1: Original UOp Graph (Single-threaded)

```python
# For: result = data.sum()
# Simplified UOp graph:

RANGE(0..1024, LOOP)  # Loop variable i: 0, 1, 2, ..., 1023
  ↓
INDEX(data, i)        # Memory access: data[i]
  ↓
LOAD(...)             # Load data[i]
  ↓
REDUCE(SUM)           # Accumulate sum
  ↓
STORE(result)
```

**Generated code:**
```c
void kernel(float* data) {
    float sum = 0;
    for (int i = 0; i < 1024; i++) {  // ← RANGE(0..1024, LOOP)
        sum += data[i];                // ← INDEX uses i
    }
    result[0] = sum;
}
```

### Step 2: Optimizer Applies THREAD Transformation

When `Opt(OptOps.THREAD, axis=0, arg=8)` is applied:

**File: [postrange.py:140](tinygrad/codegen/opt/postrange.py#L140)**
```python
opt_to_at = {
    OptOps.THREAD: AxisType.THREAD,  # ← THREAD opt changes axis type
    ...
}

# Changes the RANGE UOp:
RANGE(0..1024, LOOP)  →  RANGE(0..8, THREAD) + RANGE(0..128, LOOP)
#                         ↑               ↑
#                         Thread dim      Each thread's work
```

**What this means:**
- Original loop: 1024 iterations, single thread
- After split: 8 threads × 128 iterations each

### Step 3: Codegen Replaces THREAD Ranges with SPECIAL

**File: [gpudims.py:68-101](tinygrad/codegen/gpudims.py#L68-L101)**

```python
def add_gpudims(ctx:Renderer, s:UOp):
    # Get all ranges
    all_ranges = {x.arg[0:-1]:x for x in s_topo if x.op is Ops.RANGE}

    # Find THREAD and GLOBAL axes (line 68)
    global_dims = [x for x in all_ranges if x.arg[-1] in (AxisType.GLOBAL, AxisType.THREAD)]
    # → Finds our RANGE(0..8, THREAD)

    # Create SPECIAL UOps for thread indices (line 84)
    idxs = get_grouped_dims("gidx", global_shape, ctx.global_max)
    # → Creates SPECIAL("gidx0") with size 8

    # Substitute in the graph (line 101)
    subs = {}
    subs[RANGE(0..8, THREAD)] = SPECIAL("gidx0")
    return s.substitute(subs)  # ← Replace throughout the graph!
```

**Key insight:** All uses of `RANGE(0..8, THREAD)` are replaced with `SPECIAL("gidx0")`!

### Step 4: Memory Indexing Automatically Updated

**Before substitution:**
```
INDEX expression = (RANGE_thread * 128) + RANGE_loop
                 = (thread_idx * 128) + loop_idx
```

**After substitution:**
```
INDEX expression = (SPECIAL("gidx0") * 128) + RANGE_loop
                 = (gidx0 * 128) + loop_idx
```

**The indexing math doesn't change - only the variable!**

### Step 5: Renderer Converts SPECIAL → core_id

**ClangJIT renderer:**
```python
# In ClangRenderer (cstyle.py:230)
code_for_workitem = {"g": lambda _: "core_id"}

# SPECIAL("gidx0") becomes:
int gidx0 = core_id;  /* 8 */
```

**LLVM renderer (after PR #13781):**
```python
# In LLVMRenderer (llvmir.py:141)
(UPat(Ops.SPECIAL, name="x"),
 lambda ctx,x: f"{ctx[x]} = add i32 %core_id, 0")

# SPECIAL("gidx0") becomes:
%gidx0 = add i32 %core_id, 0
```

### Step 6: Final Generated Code

**Complete generated kernel:**
```c
void kernel(float* restrict data0, float* restrict data1, int core_id) {
    float acc = 0.0f;

    int gidx0 = core_id;  /* 8 */ ← SPECIAL("gidx0") converted

    for (int Lidx0 = 0; Lidx0 < 128; Lidx0++) {  ← RANGE(0..128, LOOP)
        // Memory index = (gidx0 * 128) + Lidx0
        //              = (core_id * 128) + Lidx0
        float val = data1[((gidx0 << 7) + Lidx0)];  ← Automatically partitioned!
        acc += val;
    }

    data0[gidx0] = acc;  ← Each thread writes to different slot
}
```

## How Each Thread Gets Different Data

**Thread execution:**
```
Thread 0 (core_id=0):
  gidx0 = 0
  Access: data1[0*128 + 0..127]     = data1[0..127]
  Write:  data0[0]

Thread 1 (core_id=1):
  gidx0 = 1
  Access: data1[1*128 + 0..127]     = data1[128..255]
  Write:  data0[1]

Thread 2 (core_id=2):
  gidx0 = 2
  Access: data1[2*128 + 0..127]     = data1[256..383]
  Write:  data0[2]

...

Thread 7 (core_id=7):
  gidx0 = 7
  Access: data1[7*128 + 0..127]     = data1[896..1023]
  Write:  data0[7]
```

**Perfect partitioning!** Each thread:
1. Reads different slice of input (no overlap)
2. Writes to different output location (no races)

## The Magic: Graph Substitution

**The genius of this design:**

1. **Index expressions are already in the graph**
   - `INDEX = (range_var * stride) + offset`
   - All the math is already computed

2. **Changing variable changes everything**
   - Replace `range_var` with `gidx0`
   - All memory accesses automatically update!

3. **No need to recompute indexing**
   - The graph transformation does it all
   - Renderer just converts to target language

## Example Trace

Let's trace a specific memory access:

**Original (single-threaded):**
```
i = RANGE(0..1024, LOOP)
index = i * 4           # Access data in chunks of 4
addr = base + index     # Final address
load(addr)              # Load from memory
```

**After THREAD transformation:**
```
# RANGE split into THREAD × LOOP
thread_idx = RANGE(0..8, THREAD)    # 8 threads
loop_idx = RANGE(0..128, LOOP)      # 128 iterations per thread
i = thread_idx * 128 + loop_idx     # Reconstruct full index
index = i * 4                        # Same math!
addr = base + index
load(addr)
```

**After SPECIAL substitution:**
```
gidx0 = SPECIAL("gidx0")            # Thread index
loop_idx = RANGE(0..128, LOOP)
i = gidx0 * 128 + loop_idx          # gidx0 replaces thread_idx
index = i * 4
addr = base + index
load(addr)
```

**Generated code:**
```c
int gidx0 = core_id;  // Thread 0: gidx0=0, Thread 1: gidx0=1, ...
for (int loop_idx = 0; loop_idx < 128; loop_idx++) {
    int i = (gidx0 * 128) + loop_idx;
    int index = i * 4;
    float val = *(base + index);
    ...
}
```

**For Thread 0 (core_id=0):**
- Loop iteration 0: i = 0*128 + 0 = 0, loads base[0..3]
- Loop iteration 1: i = 0*128 + 1 = 1, loads base[4..7]
- ...

**For Thread 1 (core_id=1):**
- Loop iteration 0: i = 1*128 + 0 = 128, loads base[512..515]
- Loop iteration 1: i = 1*128 + 1 = 129, loads base[516..519]
- ...

**Completely different data!**

## Why This Design Is Brilliant

### 1. **Automatic Correctness**
- Index math computed once in UOp graph
- Graph transformation preserves semantics
- Impossible to get indexing wrong!

### 2. **Reusable Infrastructure**
- Same mechanism for GPU (GLOBAL/LOCAL axes)
- Same mechanism for SIMD (UPCAST axes)
- Same mechanism for loop unrolling (UNROLL axes)

### 3. **Optimization Independence**
- Optimizer picks which axes to thread
- Codegen automatically handles partitioning
- No need to manually update indexing

### 4. **Verifiable**
- UOp graph is the source of truth
- Can verify transformations preserve correctness
- Can visualize with `VIZ=1`

## Summary

**Q: Is the kernel handwritten?**
**A:** NO! It's generated through graph transformations.

**Q: How does it know how to partition?**
**A:** The memory indexing uses loop variables that get replaced with thread indices.

**The flow:**
```
RANGE(LOOP) → Apply THREAD opt → RANGE(THREAD) + RANGE(LOOP)
                                        ↓
                                 Replace with SPECIAL
                                        ↓
                              All INDEX expressions updated
                                        ↓
                                Render SPECIAL → gidx0 = core_id
                                        ↓
                              Each thread gets different data!
```

**The key:** Graph substitution automatically updates all uses of the loop variable to use the thread index instead. The indexing math stays the same, but the variable changes - so each thread naturally accesses different memory!
