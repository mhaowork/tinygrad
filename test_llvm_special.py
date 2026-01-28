#!/usr/bin/env python3
"""
Test LLVM SPECIAL rendering with an operation that actually gets threaded.
"""
import os
os.environ['CPU_LLVM'] = '1'

from tinygrad import Tensor, Device
Device.DEFAULT = 'CPU'

# Element-wise operations are more likely to get threaded
# A large enough tensor should trigger threading
size = 1024 * 1024
t1 = Tensor.ones(size)
t2 = Tensor.ones(size)

# Element-wise multiply (more likely to be threaded than reduction)
result = (t1 * t2).realize()

print(f"Result shape: {result.shape}")
print(f"Sum check: {result.numpy().sum()}")  # Should be size
