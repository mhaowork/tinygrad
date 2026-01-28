# UPCAST Optimization Investigation

## Problem Statement
2048x2048 sum is 1.33x slower than PyTorch, while 4096x4096 is 0.95x faster.
The root cause is that 2048x2048 doesn't get the `UPCAST` optimization that 4096x4096 receives.

## Root Cause Analysis

### Reduction Splitting Behavior

**2048x2048**:
```
split 256: (2048, 2048) -> (8, 2048, 256) -> (1, 1)
2 kernels generated
First kernel opts: UNROLL, THREAD (no UPCAST)
```

**4096x4096**:
```
split 256: (4096, 4096) -> (16, 4096, 256) -> (1, 1)
split 16:  (16, 4096, 256) -> (1, 4096, 256, 16) -> (1, 1, 256)
3 kernels generated
First kernel opts: UPCAST, UNROLL, THREAD (has UPCAST!)
```

### Why the Difference?

The split logic divides by 256:
- **2048 / 256 = 8** → Only one split (8 < 16, no second split)
- **4096 / 256 = 16** → Gets second split (16 = threshold)

The second split creates an additional dimension, giving the kernel more axes to work with.

### UPCAST Heuristic Threshold

Location: [tinygrad/codegen/opt/heuristic.py:110](tinygrad/codegen/opt/heuristic.py#L110)

```python
while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):
    # ... upcast logic
```

The **critical condition**: `prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024`

This requires the product of upcastable dimensions in the output shape to be at least 1024.

### Why 2048 Fails the Test

For 2048x2048:
- After split 256: shape is (8, 2048, 256)
- Only gets 1 split, creating fewer upcastable dimensions
- Product of upcastable output dimensions < 1024
- UPCAST logic doesn't execute

For 4096x4096:
- After two splits: shape has more dimensions
- Product of upcastable output dimensions >= 1024
- UPCAST logic executes
- Gets `more upcast axis : [(1, 4097, 0, 4)]`

### Debug Evidence

Running with `DEBUG=4`:

**2048x2048**: No "more upcast" messages (condition fails)

**4096x4096**: Shows `more upcast axis : [(1, 4097, 0, 4)]` (condition passes)

## Potential Fixes

### Option 1: Lower the 1024 Threshold
Modify [heuristic.py:110](tinygrad/codegen/opt/heuristic.py#L110):
```python
# Current
while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):

# Proposed
while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 512) and (k.upcast_size() < 32):
```

**Pros**: Simple, may help smaller reductions
**Cons**: Could cause over-upcasting on other operations, need to benchmark

### Option 2: Adjust Split Thresholds
Make the split logic create more dimensions for mid-sized reductions.

**Pros**: More systematic fix
**Cons**: May affect many kernels, needs careful tuning

### Option 3: Special Case for Reductions
Add specific logic for reduction kernels to ensure they get UPCAST even with smaller output shapes.

**Pros**: Targeted fix for the specific problem
**Cons**: More complex, adds special case logic

### Option 4: Use Beam Search by Default
Since `BEAM=2` finds better kernels for 4096 (0.72x faster), consider using beam search for reductions.

**Pros**: Finds optimal kernels automatically
**Cons**: Slower compilation time, and actually makes 2048 worse (1.53x vs 1.33x)

## Recommended Approach

1. **Test lowering the threshold** from 1024 to 512 or 768
2. **Benchmark** on various reduction sizes (1024, 2048, 3072, 4096, 8192)
3. **Verify** it doesn't hurt other operation types (matmul, conv, etc.)

## Testing Commands

```bash
# Check if UPCAST is applied
DEBUG=4 CPU=1 CPU_LLVM=1 .venv/bin/python -c "
from tinygrad import Tensor
import numpy as np
a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()
a.sum().realize()
" 2>&1 | grep "more upcast"

# Benchmark different sizes
DEBUG=0 CPU=1 CPU_LLVM=1 BEAM=0 IGNORE_BEAM_CACHE=1 python3 test/speed/external_test_speed_v_torch.py TestSpeed.test_sum
```

## Key Code Locations

- **UPCAST heuristic**: [tinygrad/codegen/opt/heuristic.py:107-133](tinygrad/codegen/opt/heuristic.py#L107-L133)
- **Threshold condition**: [tinygrad/codegen/opt/heuristic.py:110](tinygrad/codegen/opt/heuristic.py#L110)
- **Split logic**: Likely in schedule.py or related files
- **output_shape definition**: [tinygrad/codegen/opt/postrange.py:323-324](tinygrad/codegen/opt/postrange.py#L323-L324)
- **upcastable_dims definition**: [tinygrad/codegen/opt/postrange.py:110-111](tinygrad/codegen/opt/postrange.py#L110-L111)
