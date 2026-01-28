import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner

reduce_src = r"""
// data1 is 16M inputs (4096*4096)
typedef float float4 __attribute__((aligned(16),vector_size(16)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ridx0 = 0; ridx0 < 4194304; ridx0+=4) {
    float4 val0 = *((float4*)((data1 + (ridx0 << 2))));
    float4 val1 = *((float4*)((data1 + ((ridx0+1) << 2))));
    float4 val2 = *((float4*)((data1 + ((ridx0+2) << 2))));
    float4 val3 = *((float4*)((data1 + ((ridx0+3) << 2))));
    acc0 += val0+val1+val2+val3;
  }
  *(data0 + 0) = acc0[0] + acc0[1] + acc0[2] + acc0[3];
}
"""

if __name__ == "__main__":
  np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5
  a = Tensor(np_array).realize()
  with Context():#SPLIT_REDUCEOP=0):
    GlobalCounters.reset()
    out = a.sum()
    for i, ei in enumerate(out.schedule()):
      ei.lower()
      if i == 0:
        prg_spec = replace(ei.prg.p, name="reduce", src=reduce_src, lib=None)
        ei = replace(ei, prg=CompiledRunner(prg_spec))
      ei.run()
  np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
  print(out.item())
