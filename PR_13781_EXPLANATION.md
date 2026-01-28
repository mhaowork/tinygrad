# PR #13781 Explanation: LLVM CPU Threading Support

## TL;DR

This PR **enables multi-threading for the LLVM CPU renderer**, fixing exactly the issue we discovered! It makes LLVM behave like ClangJIT with threading support.

## Current State (Before PR)

```python
class LLVMRenderer(Renderer):
    has_local = False
    global_max: tuple[int, ...] | None = None  # ← Threading DISABLED
```

Result: **Single-threaded execution only** (what you're experiencing)

## After PR #13781

```python
class LLVMRenderer(Renderer):
    has_local = False
    has_threads = True                          # ← Threading ENABLED!
    global_max = (CPU_COUNT.value, 0, 0)        # ← Max threads = CPU count
```

Result: **Multi-threaded execution enabled**

## What The PR Changes

### 1. **Imports CPU_COUNT** (Line 1)

```diff
-from tinygrad.helpers import prod, AMX
+from tinygrad.helpers import prod, AMX, CPU_COUNT
```

Needed to determine how many threads the system supports.

### 2. **Enables Threading Flags** (Lines 139-140)

```diff
 class LLVMRenderer(Renderer):
     has_local = False
-    global_max: tuple[int, ...] | None = None
+    has_threads = True
+    global_max = (CPU_COUNT.value, 0, 0)
```

**What this does:**
- `has_threads = True` → Tells the optimizer threading is available
- `global_max = (CPU_COUNT.value, 0, 0)` → Sets max threads in x-dimension
- Now the heuristic optimizer can use `OptOps.THREAD` optimization!

### 3. **Adds SPECIAL UOp Handler for Thread ID** (Line 141)

```diff
-string_rewrite = base_rewrite + PatternMatcher([(UPat(Ops.WMMA, name="wmma"), render_wmma_amx)])
+string_rewrite = base_rewrite + PatternMatcher([
+  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: f" {ctx[x]} = add {ldt(dtypes.int)} %core_id, 0"),
+  (UPat(Ops.WMMA, name="wmma"), render_wmma_amx)
+])
```

**What this does:**
- When a `SPECIAL` UOp is encountered (like `gidx0` for thread ID)
- It renders as: `%result = add i32 %core_id, 0`
- This makes the thread ID available to the kernel code
- The `add i32 %core_id, 0` is effectively just a copy (LLVM will optimize it)

### 4. **Adds core_id Parameter to Function Signature** (Lines 148-151)

```diff
 def _render_fn(self, name:str, args:list[tuple[str,DType]], kernel:list[str], prefix:list[str]|None=None) -> str:
     sargs = ", ".join([f"{ldt(dt)}{' noalias align 32' if isinstance(dt, PtrDType) else ''} {name}" for name,dt in args])
+    # Add core_id for CPU threading support
+    if self.device == "CPU" and sargs:
+        sargs += ", i32 %core_id"
+    elif self.device == "CPU":
+        sargs = "i32 %core_id"
     return "\n".join(...)
```

**What this does:**
- Every LLVM function now gets a `%core_id` parameter
- This matches how the CPU runtime passes `thread_id` to kernels
- Example: `void @kernel(float* %data0, float* %data1, i32 %core_id)`

## How It All Works Together

### Before (Single-threaded):

```
Optimizer: has_threads=False → No THREAD optimization
Codegen: No SPECIAL UOps generated → global_size=(1,1,1)
Runtime: Executes kernel once with no thread_id
```

### After (Multi-threaded):

```
1. Optimizer: has_threads=True → Applies THREAD optimization
2. Codegen: Generates SPECIAL UOps for thread indices
3. Renderer: Converts SPECIAL to LLVM core_id access
4. Runtime: Executes kernel multiple times, passing different core_id
```

## Example LLVM Code Generated

### Before PR:
```llvm
define void @r_1024_4(float* noalias align 32 %data0, float* noalias align 32 %data1) #0 {
  ; Single-threaded loop processing all data
  ...
}
```

### After PR (with threading enabled):
```llvm
define void @r_256_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0 {
  %thread_id = add i32 %core_id, 0  ; Thread index from SPECIAL UOp
  ; Each thread processes 1/N of the data based on thread_id
  ...
}
```

## Connection to Our Investigation

Remember our findings:

1. **We discovered**: LLVM has `has_threads = False` → no threading
2. **We tested**: Forcing `global_size=(4,1,1)` gave 4.3x speedup!
3. **This PR**: Enables exactly that for LLVM renderer

## What This Means For Performance

### Current (LLVM without PR):
- Time: ~1.47 ms
- Bandwidth: 45.6 GB/s
- Threads: 1

### After PR #13781 (LLVM with threading):
- Time: **~0.34 ms** (estimated based on our tests)
- Bandwidth: **~196.4 GB/s** (estimated)
- Threads: 4-8 (depending on optimization)
- **4.3x faster!**

## Caveats

Even with this PR, there's still the limitation we found:
- Threading heuristic only looks at `AxisType.LOOP`
- Doesn't (yet) apply to `AxisType.REDUCE`
- So pure reductions like `.sum()` might still not get full threading benefit

But this PR is a crucial step forward!

## Why This PR Matters

1. **Enables LLVM threading**: Brings LLVM renderer to parity with ClangJIT
2. **Better code quality**: LLVM generates faster single-threaded code than ClangJIT
3. **Best of both worlds**: LLVM optimization + multi-threading
4. **Future-proof**: Sets foundation for reduce threading when that's added

## How to Test (Once Merged)

```bash
# Instead of the slow single-threaded version:
CPU=1 CPU_LLVM=1 python your_script.py  # Before: 1.47 ms

# You'll get multi-threaded LLVM:
CPU=1 CPU_LLVM=1 python your_script.py  # After: ~0.34 ms (estimated)
```

## Summary

This PR is **exactly what we need**! It:
- ✓ Enables LLVM threading (fixes `has_threads=False`)
- ✓ Sets proper `global_max` for thread limits
- ✓ Adds `core_id` parameter handling
- ✓ Maps `SPECIAL` UOps to thread ID access
- ✓ Should give us the 4.3x speedup we demonstrated

Once this PR merges, your `CPU_LLVM=1` tests should be much faster!
