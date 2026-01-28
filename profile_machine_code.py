#!/usr/bin/env python3
"""Profile tinygrad's actual kernel execution at machine code level"""

import subprocess
import numpy as np
from tinygrad import Tensor

# Generate the kernel
print("Generating tinygrad kernel...")
a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()
result = a.sum().realize()

# Get the actual compiled library path
import os
import tempfile

# Force recompile to get the .so file
from tinygrad.helpers import DEBUG, getenv
from tinygrad.device import Device

print("\n" + "="*70)
print("KERNEL ANALYSIS")
print("="*70)

# Run with DEBUG to see kernel code
import sys
sys.stdout.flush()

code = """
from tinygrad import Tensor
import numpy as np

a = Tensor(np.random.random((2048, 2048)).astype(np.float32)).realize()

# Warmup
for _ in range(5):
    a.sum().realize()

# Profile
import time
times = []
for _ in range(20):
    start = time.perf_counter()
    result = a.sum().realize()
    times.append(time.perf_counter() - start)

import numpy as np
median_time = np.median(times) * 1000
print(f"Median time: {median_time:.3f} ms")
"""

# Run with instrumentation
print("\nRunning tinygrad kernel (will capture timings)...")
subprocess.run([
    "CPU=1", "CPU_LLVM=1", "BEAM=0",
    ".venv/bin/python", "-c", code
], shell=False)
