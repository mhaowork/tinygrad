import torch
import numpy as np

# Create array
a = torch.from_numpy(np.random.random((2048, 2048)).astype(np.float32))

# Warmup
for _ in range(10):
    result = a.sum()

# This will call into Accelerate framework's vDSP_sve or similar
import time
start = time.perf_counter()
for _ in range(100):
    result = a.sum()
end = time.perf_counter()

print(f"PyTorch sum: {(end-start)/100*1000:.2f}ms, result={result}")
print(f"PyTorch uses Accelerate framework (closed source)")
print(f"Likely vDSP_sve (vector sum) or cblas_sasum")
