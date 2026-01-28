import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner

# Optimal version: matches what Clang generates for NEON intrinsics
# - 4 accumulators for ILP
# - Process 16 floats per iteration (not 64!)
# - Compact loop body
reduce_src = r"""
// data1 is 16M inputs (4096*4096)
// Optimized to match Clang's NEON output
typedef float float4 __attribute__((aligned(16),vector_size(16)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

  int size = 4096 * 4096;

  // Process 16 floats (4 vectors) per iteration
  for (int i = 0; i < size; i += 16) {
    acc0 += data1[i + 0];  // Let compiler use ldp
    acc1 += data1[i + 4];
    acc2 += data1[i + 8];
    acc3 += data1[i + 12];
  }

  // Combine accumulators
  acc0 += acc1;
  acc2 += acc3;
  acc0 += acc2;

  *(data0 + 0) = acc0[0] + acc0[1] + acc0[2] + acc0[3];
}
"""

if __name__ == "__main__":
  np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5
  a = Tensor(np_array).realize()
  with Context(SPLIT_REDUCEOP=0):
    GlobalCounters.reset()
    out = a.sum()
    for i, ei in enumerate(out.schedule()):
      ei.lower()
      if i == 0:
        prg_spec = replace(ei.prg.p, name="reduce", src=reduce_src)
        ei = replace(ei, prg=CompiledRunner(prg_spec))
      ei.run()
  np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
  print(out.item())
