#!/usr/bin/env python3
import os, sys, time, subprocess

def _run_one():
  from tinygrad import Tensor
  from tinygrad.helpers import prod
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import CompiledRunner
  from tinygrad.uop.ops import Ops

  # Make input deterministic across subprocesses.
  Tensor.manual_seed(int(os.getenv("SEED", "0")))

  # Use integer reduction so threaded vs non-threaded results should match exactly.
  shape = (int(os.getenv("H", "4096")), int(os.getenv("W", "4096")))
  x = Tensor.randint(*shape, low=-50, high=50, dtype=dtypes.int32).realize()

  # warmup (compile + run)
  x.sum(dtype=dtypes.int64).realize()

  out = x.sum(dtype=dtypes.int64)

  # find SPECIAL in compiled kernels
  special_uops = []
  special_kernels = []
  for ei in out.schedule():
    ei = ei.lower()
    if not isinstance(getattr(ei, "prg", None), CompiledRunner): continue
    uops = ei.prg.p.uops
    sp = [u for u in uops if u.op is Ops.SPECIAL]
    if sp:
      special_uops += sp
      special_kernels.append((ei.prg.p.name, tuple(ei.prg.p.global_size)))

  # measure multiple times to reduce noise
  iters = int(os.getenv("ITERS", "50"))
  times = []
  for _ in range(iters):
    st = time.perf_counter()
    y = x.sum(dtype=dtypes.int64).realize()
    # Ensure we measure actual execution time (not just enqueue) by forcing synchronization.
    _ = y.item()
    times.append((time.perf_counter() - st) * 1e3)
  times_sorted = sorted(times)
  tm_ms = times_sorted[len(times_sorted)//2]
  bw_gbs = (prod(shape) * x.dtype.itemsize / 1e9) / (tm_ms / 1e3)

  print(f"THREADS={os.getenv('THREADS', '')} shape={shape} iters={iters} median_ms={tm_ms:.3f} bw={bw_gbs:.1f}GB/s")
  print(f"special_kernels={special_kernels}")
  if special_uops:
    print("SPECIAL uops:")
    for u in special_uops:
      print(f"  {u}")
  else:
    print("SPECIAL uops: (none)")

  print(f"result={y.item():.6f}")

def _run_child(threads_val:str):
  env = os.environ.copy()
  env["THREADS"] = threads_val
  env.setdefault("SEED", "0")
  env.setdefault("CPU", "1")
  env.setdefault("CPU_LLVM", "1")
  env.setdefault("DEBUG", "0")
  return subprocess.run([sys.executable, __file__, "--child"], env=env, check=True, capture_output=True, text=True).stdout

if __name__ == "__main__":
  if "--child" in sys.argv:
    _run_one()
    sys.exit(0)

  out0 = _run_child("0")
  out1 = _run_child("1")
  print("=== THREADS=0 ===")
  print(out0, end="" if out0.endswith("\n") else "\n")
  print("=== THREADS=1 ===")
  print(out1, end="" if out1.endswith("\n") else "\n")

  # crude verification
  assert "SPECIAL uops: (none)" in out0, "expected no SPECIAL when THREADS=0"
  assert "SPECIAL uops:" in out1 and "SPECIAL uops: (none)" not in out1, "expected SPECIAL when THREADS=1"

