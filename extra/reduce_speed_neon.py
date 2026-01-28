import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner

# Use NEON intrinsics directly for maximum control
reduce_src = r"""
#include <arm_neon.h>

// Match what standalone Clang generates
void reduce(float* restrict data0, float* restrict data1) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  int size = 4096 * 4096;

  // Process 16 floats per iteration
  for (int i = 0; i < size; i += 16) {
    acc0 = vaddq_f32(acc0, vld1q_f32(&data1[i + 0]));
    acc1 = vaddq_f32(acc1, vld1q_f32(&data1[i + 4]));
    acc2 = vaddq_f32(acc2, vld1q_f32(&data1[i + 8]));
    acc3 = vaddq_f32(acc3, vld1q_f32(&data1[i + 12]));
  }

  // Combine accumulators
  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);

  // Horizontal sum using vaddvq_f32 (ARMv8+)
  *data0 = vaddvq_f32(acc0);
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
