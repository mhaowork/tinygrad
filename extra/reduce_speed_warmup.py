import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner

reduce_src = r"""
// data1 is 16M inputs
typedef float float4 __attribute__((aligned(32),vector_size(16)));
void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc4 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc5 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc6 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc7 = {0.0f, 0.0f, 0.0f, 0.0f};
  float* data1_1 = data1+4194304;
  float* data1_2 = data1+(4194304*2);
  float* data1_3 = data1+(4194304*3);
  for (int ridx0 = 0; ridx0 < 16777216/4; ridx0+=16) {
    float4 val0 = *(float4*)((data1+(ridx0+0)));
    float4 val1 = *(float4*)((data1+(ridx0+4)));
    float4 val2 = *(float4*)((data1+(ridx0+8)));
    float4 val3 = *(float4*)((data1+(ridx0+12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    val0 = *(float4*)((data1_1+(ridx0+0)));
    val1 = *(float4*)((data1_1+(ridx0+4)));
    val2 = *(float4*)((data1_1+(ridx0+8)));
    val3 = *(float4*)((data1_1+(ridx0+12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
    val0 = *(float4*)((data1_2+(ridx0+0)));
    val1 = *(float4*)((data1_2+(ridx0+4)));
    val2 = *(float4*)((data1_2+(ridx0+8)));
    val3 = *(float4*)((data1_2+(ridx0+12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    val0 = *(float4*)((data1_3+(ridx0+0)));
    val1 = *(float4*)((data1_3+(ridx0+4)));
    val2 = *(float4*)((data1_3+(ridx0+8)));
    val3 = *(float4*)((data1_3+(ridx0+12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
  }
  float4 out = acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7;
  *(data0+0) = out[0]+out[1]+out[2]+out[3];
}
"""

if __name__ == "__main__":
  np_array = np.random.default_rng().random((2048, 2048), dtype=np.float32) - 0.5
  a = Tensor(np_array).realize()

  print("Running 10 iterations to measure cache effects...")

  with Context():#SPLIT_REDUCEOP=0):
    for iteration in range(10):
      GlobalCounters.reset()
      out = a.sum()
      for i, ei in enumerate(out.schedule()):
        ei.lower()  # populate ei.prg
        if i == 0:
          prg_spec = replace(ei.prg.p, name="reduce", src=reduce_src)
          ei = replace(ei, prg=CompiledRunner(prg_spec))
        ei.run()

      print(f"Iteration {iteration}: {out.item():.2f}")

  np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
