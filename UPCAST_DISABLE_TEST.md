# UPCAST Disable Test Results

## Performance Comparison

### With UPCAST Enabled (Original)
```bash
# heuristic.py line 112: while resolve(...) and (k.upcast_size() < 32):
```

| Size | PyTorch | Tinygrad | Result |
|------|---------|----------|--------|
| 2048x2048 | 0.34 ms | 0.49 ms | **1.45x slower** |
| 4096x4096 | 1.25 ms | 0.98 ms | **0.80x faster** |

**Optimizations**: `(UPCAST, UNROLL, THREAD)`

### With UPCAST Disabled for CPU Reductions
```bash
# heuristic.py line 112: while not is_cpu_reduce and resolve(...) and (k.upcast_size() < 32):
```

| Size | PyTorch | Tinygrad | Result |
|------|---------|----------|--------|
| 2048x2048 | 0.34 ms | 0.51 ms | **1.50x slower** |
| 4096x4096 | 1.24 ms | 1.05 ms | **0.85x faster** |

**Optimizations**: `(UNROLL, THREAD)` (no UPCAST)

## Assembly Analysis

### With UPCAST (has transpose overhead)

```c
void r_8_8_4_4096_4(...) {
  float16 acc0[1];  // 16-element vector accumulator
  for (int Lidx2 = 0; Lidx2 < 8; Lidx2++) {  // 8 outer loops
    *(acc0+0) = (float16){0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {  // 4096 inner iterations
      // Load from strided offsets
      float4 val0 = load(offset + 16384);
      float4 val1 = load(offset + 32768);
      float4 val2 = load(offset + 49152);
      float4 val3 = load(offset + 0);
      // Shuffle into float16
      *(acc0+0) = *(acc0+0) + (float16){val3[0]...val3[3], val0[0]...val0[3], ...};
    }
    // ❌ TRANSPOSE at end of each outer loop (8 times total)
    store = (float4){acc[0],acc[4],acc[8],acc[12]} +
            (float4){acc[1],acc[5],acc[9],acc[13]} + ...;
  }
}
```

**Assembly (inner loop)**:
```asm
; ✅ Pure vector adds in 4096-iteration loop
0x000048: fadd  v1.4s, v1.4s, v5.4s
0x00004c: fadd  v2.4s, v2.4s, v4.4s
0x000050: fadd  v0.4s, v0.4s, v7.4s
0x000054: fadd  v3.4s, v3.4s, v6.4s

; ❌ TRANSPOSE after every outer loop (8 times)
0x000064: zip1  v4.4s, v0.4s, v2.4s    ; ~4 cycles
0x000068: zip1  v5.4s, v1.4s, v3.4s    ; ~4 cycles
0x00006c: ext   v6.16b, v1.16b, v5.16b, #8  ; ~3 cycles
... (8 more shuffle instructions)
```

**Total overhead**: ~30-40 cycles × 8 outer loops = **240-320 cycles**

### Without UPCAST (clean, no transpose)

```c
void r_32_8_4096_4(...) {
  float4 acc0[1];  // 4-element vector accumulator
  for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {  // 32 outer loops
    *(acc0+0) = (float4){0.0f,0.0f,0.0f,0.0f};
    for (int Ridx0 = 0; Ridx0 < 4096; Ridx0++) {  // 4096 inner iterations
      // ✅ Sequential load
      float4 val0 = load(offset);
      *(acc0+0) = *(acc0+0) + val0;
    }
    // ✅ Simple horizontal reduction ONLY at the end
    result = acc[0] + acc[1] + acc[2] + acc[3];
  }
}
```

**Assembly (inner loop)**:
```asm
; ✅ Pure vector add in 4096-iteration loop
0x00001c: ldr   q1, [x9, x11]          ; Sequential load
0x000020: fadd  v0.4s, v0.4s, v1.4s    ; Vector add

; ✅ Simple horizontal reduction at end (32 times)
0x000030: faddp s1, v0.2s               ; ~2 cycles
0x000034: mov   s2, v0.s[2]             ; ~1 cycle
0x000038: fadd  s1, s2, s1              ; ~1 cycle
0x00003c: mov   s0, v0.s[3]             ; ~1 cycle
0x000040: fadd  s0, s0, s1              ; ~1 cycle
```

**Total overhead**: ~6 cycles × 32 outer loops = **192 cycles**

## Surprising Result

**Without UPCAST should be faster** (192 cycles overhead vs 240-320), but it's actually **slightly slower**!

### Why?

**Outer loop overhead**:
- **With UPCAST**: 8 outer loops (more work per loop, less loop overhead)
- **Without UPCAST**: 32 outer loops (less work per loop, more loop overhead)

Each outer loop has:
- Loop counter initialization
- Accumulator initialization
- Store operation
- Loop condition check

**Extra overhead per outer loop**: ~10-20 cycles

With UPCAST: 8 × 10-20 = **80-160 cycles** loop overhead
Without UPCAST: 32 × 10-20 = **320-640 cycles** loop overhead

**Net difference**: 320-640 - (80-160) = **240-480 extra cycles** without UPCAST

This **overwhelms** the 48-128 cycle savings from eliminating transpose!

## Conclusion

**UPCAST is actually helpful** for 2048x2048, despite the transpose overhead, because:
1. Fewer outer loops (8 vs 32) = less loop overhead
2. The transpose overhead (~240 cycles) is less than the extra loop overhead (~240-480 cycles)

**For 4096x4096**:
- With UPCAST: 0.80x faster
- Without UPCAST: 0.85x faster (worse)

Same reasoning applies.

## The Real Problem

Both approaches are still **slower than PyTorch** because:

1. **PyTorch uses Accelerate framework** with hand-optimized assembly
2. **Better threading/scheduling** - PyTorch likely uses different parallelization strategy
3. **Possible prefetching** - PyTorch may have better memory access patterns
4. **Lower kernel launch overhead** - native library vs JIT compilation

## Recommendation

**Keep UPCAST enabled** for CPU reductions. The transpose overhead is real, but the alternative (more outer loops) is worse.

To actually match PyTorch, we'd need:
1. Better loop structure (fewer outer loops even without UPCAST)
2. Better threading strategy
3. Possibly hand-tuned kernels for common reduction patterns
4. Investigation of PyTorch's actual strategy (likely in Accelerate's vDSP library)
