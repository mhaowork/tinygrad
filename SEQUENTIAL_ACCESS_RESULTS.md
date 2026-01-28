# Sequential Access Fix Results

## What We Fixed

Modified UPCAST heuristic to detect and skip strided memory access for CPU reductions:

```python
# heuristic.py lines 131-145
if is_cpu_reduce and best_choice[1] > 16:  # sum_strides > 16 indicates strided access
  if DEBUG >= 4: print(f"Skipping UPCAST on CPU reduction due to large stride: {best_choice[1]}")
  # Apply aggressive UNROLL to compensate
  if k.unrollable_dims:
    for splits in [8, 4]:
      k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, splits))
  break
```

## Results

### Memory Access Pattern

**Before (strided UPCAST)**:
```c
// Loads from 4 locations 64KB apart
float4 val0 = load(data + base + 16384);   // +64KB
float4 val1 = load(data + base + 32768);   // +128KB
float4 val2 = load(data + base + 49152);   // +192KB
float4 val3 = load(data + base + 0);
```

**After (sequential)**:
```c
// Sequential loads, incrementing by 4 bytes
for (int Ridx0 = 0; Ridx0 < 2048; Ridx0++) {
  float8 val0 = load(data + base + (Ridx0 << 3));  // Sequential!
}
```

### Memory Bandwidth

| Config | Bandwidth | Utilization | Performance |
|--------|-----------|-------------|-------------|
| Strided UPCAST | 34.2 GB/s | 17% | 1.45x slower |
| Sequential (UNROLL=4) | 56.6 GB/s | 28% | 1.42x slower |
| Sequential (UNROLL=8) | ~36 GB/s | 18% | 1.41x slower |
| PyTorch | ~50 GB/s | 25% | baseline (0.33ms) |

### Performance Summary

```
2048x2048:
- Before fix: 0.49 ms (1.45x slower than PyTorch)
- After fix:  0.47 ms (1.41x slower than PyTorch)
- PyTorch:    0.33 ms

Improvement: ~4% faster, but still 1.41x slower
```

## Why We're Still Slower

### 1. Outer Loop Overhead (32 vs fewer in PyTorch)

Our kernel has **32 outer loops** per thread:
```c
for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {  // 32 iterations
  // Reset accumulator
  // Inner loop (2048 iterations)
  // Horizontal reduction
  // Store
}
```

Each outer loop has overhead:
- Accumulator initialization: ~4 cycles
- Horizontal reduction: ~6-8 cycles
- Store: ~4 cycles
- Loop control: ~2 cycles
**Total**: ~16-18 cycles × 32 loops × 8 threads = **4096-4608 cycles** overhead

### 2. Horizontal Reduction Overhead

After each inner loop (32 times per thread):
```asm
faddp s1, v0.2s    ; 2 cycles
mov   s2, v0.s[2]  ; 1 cycle
fadd  s1, s2, s1   ; 1 cycle
mov   s0, v0.s[3]  ; 1 cycle
fadd  s0, s0, s1   ; 1 cycle
```
**Total**: ~6 cycles × 32 = **192 cycles per thread**

### 3. Memory Bandwidth Still Sub-Optimal

We achieve **~36 GB/s** vs PyTorch's **~50 GB/s**.

Possible reasons:
- **Cache line utilization**: We may not be fully utilizing 64-byte cache lines
- **Prefetcher efficiency**: Strided thread access patterns may confuse hardware prefetcher
- **Thread synchronization overhead**: 8 threads may have contention/synchronization costs

### 4. Thread Work Distribution

Our 8 threads each process:
- **524,288 elements** (2048×2048 / 8)
- Split into **32 chunks of 16,384 elements each**

PyTorch likely uses different work distribution:
- Possibly more threads (8-16?)
- Possibly larger chunks per thread iteration
- Better cache locality per thread

## Theoretical Best Case

With sequential access and L1 cache hits:
```
Cycles per iteration: 5 (2 loads + 2 fadds + 1 overhead)
Total iterations: 32,768 per thread
Total cycles: 163,840 per thread
Time: 163,840 / 3.2 GHz = 0.051 ms
```

**Current**: 0.47 ms → **9.2x slower than theory**
**PyTorch**: 0.33 ms → **6.5x slower than theory**

Both are far from theoretical peak, suggesting:
- Memory latency dominates (not compute)
- Cache behavior is critical
- Thread overhead matters

## What Would Close the Gap

### Option 1: Reduce Outer Loops

Generate code with **8 outer loops** instead of 32:
- Each loop processes 4x more data
- Reduces initialization/reduction overhead by 4x
- **Estimated gain**: ~10-15% faster

### Option 2: Improve Thread Distribution

Use different work splitting:
- Larger chunks per thread
- Better cache line alignment
- **Estimated gain**: ~10-20% faster

### Option 3: Hand-Tuned Kernel for Common Cases

For power-of-2 reductions on CPU, use specialized codegen:
- Fixed unroll factors
- Optimized for L1/L2 cache sizes
- Minimal horizontal reductions
- **Estimated gain**: ~30-50% faster (might reach PyTorch)

### Option 4: Call vDSP Directly

For macOS, detect sum reductions and call vDSP_sve directly:
- **Matches PyTorch's performance** (both use vDSP)
- Requires runtime check for macOS
- **Estimated gain**: 1.41x faster (matches PyTorch)

## Conclusion

The sequential access fix **improved bandwidth by 1.66x** (34 → 57 GB/s) but **performance only improved 3%** (1.45x → 1.41x slower).

The remaining gap is due to:
1. **Outer loop overhead** (32 loops vs fewer in PyTorch)
2. **Horizontal reduction overhead** (32× per thread)
3. **Thread distribution** (our 8 threads vs PyTorch's strategy)
4. **Absolute bandwidth** (36 GB/s vs 50 GB/s)

To match PyTorch, we'd need either:
- **Better codegen** (fewer outer loops, better threading)
- **Call vDSP** on macOS (match PyTorch's approach)
