import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner

# Better version: accumulate directly without intermediate temps
reduce_src = r"""
// data1 is 16M inputs (4096*4096)
typedef float float4 __attribute__((aligned(16),vector_size(16)));

void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int ridx0 = 0; ridx0 < 4194304; ridx0+=16) {
    // Accumulate directly to break dependency chains
    acc0 += *((float4*)((data1 + ((ridx0+0) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+1) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+2) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+3) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+4) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+5) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+6) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+7) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+8) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+9) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+10) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+11) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+12) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+13) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+14) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+15) << 2))));
  }

  // Combine accumulators
  float4 total = acc0 + acc1 + acc2 + acc3;
  *(data0 + 0) = total[0] + total[1] + total[2] + total[3];
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
