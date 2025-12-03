
from extra.gemm.amd_uop_matmul import copy
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes  # tensor API and perf counters
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, KernelInfo, sint, AxisType         # UOp IR nodes and loop axis helpers
from tinygrad.engine.realize import ExecItem, get_runner             # turn a UOp kernel into a runnable exec item
from tinygrad.dtype import AddrSpace                                 # address spaces: GLOBAL, LOCAL (shared), REG
from tinygrad.helpers import getenv  


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

THREADS_PER_BLOCK = 128
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
  blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")  
  blockIdx_y = UOp.special(M // BLOCK_M, "gidx1")
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  a = UOp.placeholder((N, N), dtypes.half, slot=1) # input A
  b = UOp.placeholder((N, N), dtypes.half, slot=2) # input B
  c = UOp.placeholder((N, N), dtypes.half, slot=0) # output C

  # TODO(mhao): bank conflicts?
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

  # Load from shared to registers for this warp/thread
  # Exact indexing depends on NVIDIA's 16x8x16 layout
  warpIdx = (tid // WARP_SIZE) % WARPS_IN_BLOCK_X
  warpIdy = (tid // WARP_SIZE) // WARPS_IN_BLOCK_X

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WARP_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WARP_X

  A_frag = UOp.placeholder((ITERS_PER_WARP_M, TM), dtypes.half, slot=0, addrspace=AddrSpace.REG)  # per-thread A fragment
  A_frag = copy(A_frag, As[k, :].reshape(WARPS_IN_BLOCK_Y, ITERS_PER_WARP_M, LANES_PER_WARP_Y, TM)[warpIdy, :, laneIdy, :], 300, set=True, upcast=True)


  B_frag = UOp.placeholder((ITERS_PER_WARP_N, TN), dtypes.half, slot=1, addrspace=AddrSpace.REG)  # per-thread B fragment
  B_frag = copy(B_frag, Bs[k, :].reshape(WARPS_IN_BLOCK_X, ITERS_PER_WARP_N, LANES_PER_WARP_X, TN)[warpIdx, :, laneIdx, :], 400, set=True, upcast=True)
  
  wmma_arg = (
    'WMMA_16_8_16_half_float',  # name
    (16, 8, 16),                 # (M, N, K) dimensions
    dtypes.half,                 # input dtype
    dtypes.float,                # accumulator dtype
    'CUDA',                      # device
    32,                          # threads per warp
    upcast_axes,                 # TBD: match NVIDIA layout
    ()                           # extra args
  )
  out = UOp(Ops.WMMA, dtypes.float.vec(4), (A_frag, B_frag, c_regs_load), arg=wmma_arg)

