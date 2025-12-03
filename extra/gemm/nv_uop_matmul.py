
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from extra.gemm.amd_uop_matmul import copy, rngs_for_shape, test_matmul

N = getenv("N", 4096)
M = K = N # square

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

  As_store = copy(As.reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)
  Bs_store = copy(Bs.reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  # TODO: can we automate barrier? (similarly in amd_uop_matmul)
  barrier = UOp.barrier(As_store, Bs_store)
  As, Bs = As.after(barrier), Bs.after(barrier)  # thread-safe views

  # open inner k range
  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)  # loop over K slice within this block tile

  # ---------------------------
  # LOCAL -> REG (per-warp tiles)
  # ---------------------------
  warpIdx = (tid // WARP_SIZE) % WARPS_IN_BLOCK_X
  warpIdy = (tid // WARP_SIZE) // WARPS_IN_BLOCK_X
  assert warpIdy.vmax+1 == WARPS_IN_BLOCK_Y

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WARP_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WARP_X
  assert laneIdy.vmax+1 == LANES_PER_WARP_Y

  A_frag = UOp.placeholder((ITERS_PER_WARP_M, TM), dtypes.half, slot=0, addrspace=AddrSpace.REG)  # per-thread A fragment
  A_frag = copy(A_frag, As[k, :].reshape(WARPS_IN_BLOCK_Y, ITERS_PER_WARP_M, LANES_PER_WARP_Y, TM)[warpIdy, :, laneIdy, :], 300, set=True, upcast=True)

  B_frag = UOp.placeholder((ITERS_PER_WARP_N, TN), dtypes.half, slot=1, addrspace=AddrSpace.REG)  # per-thread B fragment
  B_frag = copy(B_frag, Bs[k, :].reshape(WARPS_IN_BLOCK_X, ITERS_PER_WARP_N, LANES_PER_WARP_X, TN)[warpIdx, :, laneIdx, :], 400, set=True, upcast=True)
  
  # ---------------------------
  # FMA: c_regs (fp32) += A_frag (half) * B_frag (half)
  # TODO: wmma??
  # ---------------------------
  c_regs = UOp.placeholder((ITERS_PER_WARP_M, TM, ITERS_PER_WARP_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)  # accumulators
  i = UOp.range(c_regs.size, 16)  # flat init loop
  c_regs = c_regs.after(c_regs.flatten()[i].store(0.0).end(i))

  # accumulate: c_regs += A_frag * B_frag
  iterWarpM, yt, iterWarpN, xt = rngs = rngs_for_shape(c_regs.shape, 500)
  sink = c_regs[*rngs].store(c_regs.after(k)[*rngs] + A_frag[iterWarpM, yt].cast(dtypes.float) * B_frag[iterWarpN, xt].cast(dtypes.float)).end(iterWarpM, iterWarpN, yt, xt)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape(WARPS_IN_BLOCK_Y, ITERS_PER_WARP_M, LANES_PER_WARP_Y, TM,
                WARPS_IN_BLOCK_X, ITERS_PER_WARP_N, LANES_PER_WARP_X, TN)
  c = c[warpIdy, :, laneIdy, :,
        warpIdx, :, laneIdx, :]  # pick this thread's output tile
  sink = copy(c, c_regs.after(sink).cast(dtypes.half), rng=600)  # write accumulators back to global

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

if __name__ == "__main__":
  test_matmul(hand_spec(), N=N)

