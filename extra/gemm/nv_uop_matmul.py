
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from extra.gemm.amd_uop_matmul import copy, rngs_for_shape

N = getenv("N", 4096)
M = K = N # square
run_count = getenv("CNT", 5)
USE_WMMA = getenv("WMMA", 1)

WARP_SIZE = 32
BLOCK_M = 64    # rows of C per block
BLOCK_N = 128   # columns of C per block  
BLOCK_K = 64    # K-slice per iteration
THREADS_PER_BLOCK = 128  # 4 warps

# NV 16x8x16 tensor core
TC_M = 16
TC_N = 8
TC_K = 16

# Reg tile sizse
TM = 2  # rows per thread
TN = 8  # columns per thread

assert THREADS_PER_BLOCK % BLOCK_N == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_N"
assert THREADS_PER_BLOCK % BLOCK_K == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_K"
assert (BLOCK_N * BLOCK_K) % THREADS_PER_BLOCK == 0
assert (BLOCK_M * BLOCK_K) % THREADS_PER_BLOCK == 0

WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
WARP_TILE_N = 128
WARP_TILE_M = BLOCK_N * BLOCK_M // WARPS_PER_BLOCK // WARP_TILE_N
assert BLOCK_N % WARP_TILE_N == 0, "BN must be a multiple of WN"
assert BLOCK_M % WARP_TILE_M == 0, "BM must be a multiple of WM"
WARPS_IN_BLOCK_X = BLOCK_N // WARP_TILE_N
WARPS_IN_BLOCK_Y = BLOCK_M // WARP_TILE_M
assert WARPS_IN_BLOCK_X * WARPS_IN_BLOCK_Y == WARPS_PER_BLOCK, "warp grid must match warps/block"

LANES_PER_WARP_X = 8
LANES_PER_WARP_Y = 4
ITERS_PER_WARP_N = WARP_TILE_N // (LANES_PER_WARP_X * TN)
ITERS_PER_WARP_M = WARP_TILE_M // (LANES_PER_WARP_Y * TM)
assert WARP_TILE_N % (LANES_PER_WARP_X * TN) == 0, "WARP_TILE_N must be divisible by LANES_PER_WAVE_X*TN"
assert WARP_TILE_M % (LANES_PER_WARP_Y * TM) == 0, "WARP_TILE_M must be divisible by LANES_PER_WAVE_Y*TM"


def hand_spec():
  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")   # grid x covers N dimension tiles
  blockIdx_y = UOp.special(M // BLOCK_M, "gidx1")   # grid y covers M dimension tiles

  a = UOp.placeholder((N, N), dtypes.half, slot=1) # input A
  b = UOp.placeholder((N, N), dtypes.half, slot=2) # input B
  c = UOp.placeholder((N, N), dtypes.half, slot=0) # output C

  # index the output with the globals (select this block's C tile)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[blockIdx_y, :, blockIdx_x, :]

  # open the main reduction range over K tiles, slice A/B accordingly
  k_tile_range = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)
  a = a.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_K, BLOCK_K)[blockIdx_y, :, k_tile_range, :]
  b = b.reshape(N // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, blockIdx_x, :]

  # globals are no longer used, they are already in the indexes
  del blockIdx_y, blockIdx_x

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")  # thread id within block

  # TODO: bank conflicts? May need padding like AMD kernel
  As = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

  As_store = copy(As.permute((1, 0)).reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)
  Bs_store = copy(Bs.reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  # TODO: can we automate barrier? (similarly in amd_uop_matmul)
  barrier = UOp.barrier(As_store, Bs_store)
  As, Bs = As.after(barrier), Bs.after(barrier)  # thread-safe views

  warpIdx = (tid // WARP_SIZE) % WARPS_IN_BLOCK_X
  warpIdy = (tid // WARP_SIZE) // WARPS_IN_BLOCK_X
  assert warpIdy.vmax+1 == WARPS_IN_BLOCK_Y

  lane = tid % WARP_SIZE
  laneIdx = lane % LANES_PER_WARP_X
  laneIdy = lane // LANES_PER_WARP_X
  assert laneIdy.vmax+1 == LANES_PER_WARP_Y

  # USE_WMMA:
  # ---------------------------
  # tensor core path (m16n8k16)
  # ---------------------------
  # accumulators: one vec4 per 8-column tile in this warp
  c_regs = UOp.placeholder((WARP_TILE_N // TC_N,), dtypes.float.vec(4), slot=2, addrspace=AddrSpace.REG)
  init_i = UOp.range(c_regs.size, 16)
  c_regs = c_regs.after(c_regs[init_i].store(UOp.const(c_regs.dtype, 0.0)).end(init_i))

  k_block = UOp.range(BLOCK_K // TC_K, 3, AxisType.REDUCE)
  k_base = k_block * TC_K

  row_mod8 = lane // 4
  k_half = lane % 4
  a_row_base = warpIdy * TC_M

  # build A fragment (8 halfs per thread) from shared As tile
  A_frag = UOp.placeholder((8,), dtypes.half, slot=0, addrspace=AddrSpace.REG)
  a_idx = UOp.range(8, 300, AxisType.UPCAST)
  a_row_hi = (a_idx // 2) % 2
  a_k_hi = a_idx // 4
  a_parity = a_idx % 2
  a_row = row_mod8 + a_row_hi * 8
  a_k = a_k_hi * 8 + k_half * 2 + a_parity
  A_frag = A_frag.after(A_frag[a_idx].store(As[k_base + a_k, a_row_base + a_row]).end(a_idx))

  # loop N tiles (16x8 each) and issue WMMA
  n_tile = UOp.range(WARP_TILE_N // TC_N, 4)
  B_frag = UOp.placeholder((4,), dtypes.half, slot=1, addrspace=AddrSpace.REG)
  b_idx = UOp.range(4, 310, AxisType.UPCAST)
  b_k_hi = b_idx // 2
  b_parity = b_idx % 2
  b_k = b_k_hi * 8 + k_half * 2 + b_parity
  b_col = lane // 4
  B_frag = B_frag.after(B_frag[b_idx].store(Bs[k_base + b_k, n_tile * TC_N + b_col]).end(b_idx))

  wmma_arg = ('WMMA_8_16_16_half_float', (8, 16, 16), dtypes.half, dtypes.float, 'CUDA', WARP_SIZE,
              (((0, 2), (1, 2), (2, 2)), ((0, 2), (1, 2)), ((0, 2), (1, 2))), ())
  wmma_out = UOp(Ops.WMMA, dtypes.float.vec(4), (A_frag, B_frag, c_regs.after(B_frag)[n_tile]), arg=wmma_arg)
  sink = c_regs.after(wmma_out)[n_tile].store(wmma_out).end(n_tile)

  # Close k_block, sync, and close K tiles
  sink = sink.end(k_block).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  n_idx = UOp.range(WARP_TILE_N // TC_N, 600)
  elem = UOp.range(4, 601, AxisType.UPCAST)
  out_row = warpIdy * TC_M + (lane // 4) + (elem // 2) * 8
  out_col = n_idx * TC_N + (elem % 2) + (lane % 4) * 2
  c_tile = c.reshape(BLOCK_M, BLOCK_N)[out_row, out_col]
  sink = c_tile.store(c_regs.after(sink)[n_idx].gep(elem).cast(dtypes.half)).end(n_idx, elem)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()



def test_matmul_fp16(sink:UOp, N=N):
  rng = np.random.default_rng()
  a = Tensor((rng.random((N, N), dtype=np.float32)-0.5).astype(np.float16))
  b = Tensor((rng.random((N, N), dtype=np.float32)-0.5).astype(np.float16))
  hc = Tensor.empty(N, N, dtype=dtypes.half)
  Tensor.realize(a, b, hc)

  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in [hc, a, b]])

  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a.cast(dtypes.float) @ b.cast(dtypes.float)).realize()
    with Context(DEBUG=0):
      err = (hc.cast(dtypes.float) - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-02:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  test_matmul_fp16(hand_spec(), N=N)
