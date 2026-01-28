import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import KernelInfo

if __name__ == "__main__":
  np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5
  a = Tensor(np_array).realize()
  with Context(SPLIT_REDUCEOP=0):
    GlobalCounters.reset()
    out = a.sum()
    sis = out.schedule()

    for i, ei in enumerate(sis):
      ei.lower()
      if i == 0:
        print("Original applied_opts:", ei.prg.p.ast.arg.applied_opts if ei.prg.p.ast.arg else None)

        # Try to modify the unroll amount
        # The AST has a KernelInfo with applied_opts
        if ei.prg.p.ast.arg:
          original_opts = ei.prg.p.ast.arg.applied_opts
          # Replace UNROLL opt with smaller value
          new_opts = []
          for opt in original_opts:
            if opt.op == OptOps.UNROLL:
              # Change from arg=4 to arg=2 (less unrolling)
              new_opt = Opt(op=OptOps.UNROLL, axis=opt.axis, arg=2)
              print(f"Changing {opt} to {new_opt}")
              new_opts.append(new_opt)
            else:
              new_opts.append(opt)

          # Rebuild the AST with new KernelInfo
          new_kernel_info = replace(ei.prg.p.ast.arg, applied_opts=tuple(new_opts))
          new_ast = ei.prg.p.ast.replace(arg=new_kernel_info)

          # Rebuild program spec with new AST
          new_prg_spec = replace(ei.prg.p, ast=new_ast)

          # Recompile
          new_prg = CompiledRunner(new_prg_spec)
          ei = replace(ei, prg=new_prg)

          print("New applied_opts:", ei.prg.p.ast.arg.applied_opts)

      ei.run()

  np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
  print(out.item())
