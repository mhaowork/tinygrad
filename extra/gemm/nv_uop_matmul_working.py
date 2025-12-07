"""
Working NVIDIA WMMA kernel in amd_uop_matmul style
Adapted from amd_uop_matmul.py with WMMA tensor core operations

Target: 165+ TFLOPS on RTX 4090 with FP16 input, FP32 accumulator
"""
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from extra.gemm.amd_uop_matmul import copy, rngs_for_shape

N = getenv("N", 4096)
M = K = N
run_count = getenv("CNT", 5)

# NVIDIA m16n8k16 tensor core configuration
WARP_SIZE = 32

# Block sizes (tuned for A100/4090)
BLOCK_N = 128   # columns of C per block
BLOCK_M = 128   # rows of C per block
BLOCK_K = 32    # K-slice per block iteration

# Tensor core dimensions
TC_M = 16
TC_N = 8
TC_K = 16

# Per-thread register tiles (not used directly with WMMA, but for structure)
TN = 4
TM = 4

THREADS_PER_BLOCK = 256  # 8 warps
assert THREADS_PER_BLOCK % BLOCK_N == 0
assert THREADS_PER_BLOCK % BLOCK_K == 0
assert (BLOCK_N * BLOCK_K) % THREADS_PER_BLOCK == 0
assert (BLOCK_M * BLOCK_K) % THREADS_PER_BLOCK == 0

WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
WARP_TILE_N = 64
WARP_TILE_M = BLOCK_N * BLOCK_M // WARPS_PER_BLOCK // WARP_TILE_N
assert BLOCK_N % WARP_TILE_N == 0
assert BLOCK_M % WARP_TILE_M == 0
WARPS_IN_BLOCK_X = BLOCK_N // WARP_TILE_N
WARPS_IN_BLOCK_Y = BLOCK_M // WARP_TILE_M
assert WARPS_IN_BLOCK_X * WARPS_IN_BLOCK_Y == WARPS_PER_BLOCK

# How many 16x8 tiles per warp
M_TILES = WARP_TILE_M // TC_M  # 32 / 16 = 2
N_TILES = WARP_TILE_N // TC_N  # 64 / 8 = 8


def hand_spec_nv_wmma():
    """
    NVIDIA WMMA kernel following amd_uop_matmul.py structure.

    Key differences from AMD version:
    1. Uses WMMA ops instead of scalar FMA
    2. Fragments are vec(8) for A, vec(4) for B, vec(4) for C
    3. Block K size is 32 (must be multiple of TC_K=16)
    """

    # ---------------------------
    # Block indices & placeholders
    # ---------------------------
    blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")
    blockIdx_y = UOp.special(M // BLOCK_M, "gidx1")

    a = UOp.placeholder((N, N), dtypes.half, slot=1)
    b = UOp.placeholder((N, N), dtypes.half, slot=2)
    c = UOp.placeholder((N, N), dtypes.float, slot=0)

    # Index the output with the globals
    c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[blockIdx_y, :, blockIdx_x, :]

    # Open the main reduction range
    k_tile_range = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)
    a = a.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_K, BLOCK_K)[blockIdx_y, :, k_tile_range, :]
    b = b.reshape(N // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, blockIdx_x, :]

    del blockIdx_y, blockIdx_x

    # ---------------------------
    # GLOBAL -> LOCAL (As, Bs)
    # ---------------------------
    tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

    # Shared memory (same as AMD version)
    As = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
    Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

    As_store = copy(As.permute((1, 0)).reshape(-1, THREADS_PER_BLOCK)[:, tid],
                    a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)
    Bs_store = copy(Bs.reshape(-1, THREADS_PER_BLOCK)[:, tid],
                    b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

    barrier = UOp.barrier(As_store, Bs_store)
    As, Bs = As.after(barrier), Bs.after(barrier)

    # ---------------------------
    # Warp/lane indexing
    # ---------------------------
    warpIdx = (tid // WARP_SIZE) % WARPS_IN_BLOCK_X
    warpIdy = (tid // WARP_SIZE) // WARPS_IN_BLOCK_X
    assert warpIdy.vmax+1 == WARPS_IN_BLOCK_Y

    lane = tid % WARP_SIZE

    # Decompose lane ID into bits (matching generated code pattern)
    # This is the key to proper WMMA fragment mapping
    l0 = lane & 1           # Bit 0
    l1 = (lane >> 1) & 1    # Bit 1
    l2 = (lane >> 2) & 1    # Bit 2
    l3 = (lane >> 3) & 1    # Bit 3
    l4 = lane >> 4          # Bits 4+ (0 or 1 for warp of 32)

    # ---------------------------
    # Reshape for WMMA access
    # ---------------------------
    # Reshape shared memory into TC_M x TC_N x TC_K tiles
    # As: [BLOCK_K, BLOCK_M] -> [BLOCK_K/TC_K, TC_K, BLOCK_M/TC_M, TC_M]
    As_tc = As.reshape(BLOCK_K // TC_K, TC_K, BLOCK_M // TC_M, TC_M)
    # Bs: [BLOCK_K, BLOCK_N] -> [BLOCK_K/TC_K, TC_K, BLOCK_N/TC_N, TC_N]
    Bs_tc = Bs.reshape(BLOCK_K // TC_K, TC_K, BLOCK_N // TC_N, TC_N)

    # ---------------------------
    # Initialize accumulators
    # ---------------------------
    # Each warp computes M_TILES x N_TILES tiles
    # Each tile accumulator has 4 fp32 values
    # Store as individual floats, not vec(4), for easier indexing
    c_regs = UOp.placeholder((M_TILES, N_TILES, 4), dtypes.float, slot=2, addrspace=AddrSpace.REG)

    init_i = UOp.range(c_regs.size, 16)
    c_regs = c_regs.after(c_regs.flatten()[init_i].store(UOp.const(dtypes.float, 0.0)).end(init_i))

    # ---------------------------
    # Inner K loop (iterate over TC_K chunks)
    # ---------------------------
    k_chunk = UOp.range(BLOCK_K // TC_K, 3, AxisType.REDUCE)

    # ---------------------------
    # Load fragments and do WMMA
    # ---------------------------
    # CORRECT fragment loading based on m16n8k16 WMMA layout
    # References:
    # - NVIDIA WMMA docs: each thread holds 8 elements for A (m16xk16)
    # - tc.py:78-81: swizzle patterns for WMMA_8_16_16_half_float
    # - uop_graph_wmma_generated.txt: actual bit patterns from TC optimizer

    wmma_stores = []

    # Iterate over all M_TILES x N_TILES
    for m_tile in range(M_TILES):
        # Load A fragment for this M tile (m16xk16)
        # Each thread gets 8 fp16 values from a 16x16 tile
        # Layout: threads are mapped to rows using l2,l3,l4 and columns using l0,l1
        a_tile_idx = warpIdy * M_TILES + m_tile

        a_frag_elems = []
        for elem in range(8):
            # For m16xk16:
            # - Row index within tile: base determined by l4,l3,l2 + offset by elem
            # - Col index within tile: determined by l1,l0 + offset by elem
            # Pattern from tc.py swizzle: 8 elements arranged in 2x4 pattern per thread
            row_base = (l4 * 8 + l3 * 4 + l2 * 2)  # 16 rows / 32 threads = groups
            col_idx = (elem // 4) * 8 + l1 * 4 + l0 * 2 + (elem % 4) // 2
            row_idx = row_base + (elem % 2)

            a_elem = As_tc[k_chunk, col_idx, a_tile_idx, row_idx]
            a_frag_elems.append(a_elem)

        a_frag = UOp.vectorize(*a_frag_elems)

        for n_tile in range(N_TILES):
            # Load B fragment for this N tile (k16xn8)
            # Each thread gets 4 fp16 values from a 16x8 tile
            b_tile_idx = warpIdx * N_TILES + n_tile

            b_frag_elems = []
            for elem in range(4):
                # For k16xn8:
                # - Row index (K): determined by elem and lane bits
                # - Col index (N): determined by l4,l3,l2
                # Pattern: 4 elements per thread across K and N dimensions
                row_idx = elem * 4 + l1 * 2 + l0
                col_idx = l4 * 4 + l3 * 2 + l2

                b_elem = Bs_tc[k_chunk, row_idx, b_tile_idx, col_idx]
                b_frag_elems.append(b_elem)

            b_frag = UOp.vectorize(*b_frag_elems)

            # WMMA operation
            # Arguments from tc.py:78-81 for m16n8k16 fp16->fp32
            wmma_arg = ('WMMA_8_16_16_half_float', (8, 16, 16), dtypes.half, dtypes.float, 'CUDA', WARP_SIZE,
                        (((0, 2), (1, 2), (2, 2)), ((0, 2), (1, 2)), ((0, 2), (1, 2))), ())

            # Vectorize the 4 accumulator values for WMMA input
            acc_in = UOp.vectorize(*[c_regs.after(a_frag).after(b_frag)[m_tile, n_tile, i] for i in range(4)])
            wmma_out = UOp(Ops.WMMA, dtypes.float.vec(4), (a_frag, b_frag, acc_in), arg=wmma_arg)

            # Unpack the vec(4) result and store 4 individual floats
            for i in range(4):
                wmma_stores.append(c_regs[m_tile, n_tile, i].store(wmma_out.gep(i)))

    sink = UOp.group(*wmma_stores).end(k_chunk).barrier().end(k_tile_range)

    # ---------------------------
    # Epilogue: REG -> GLOBAL
    # ---------------------------
    # CORRECT output mapping for m16n8k16 C fragment
    # Each thread has 4 fp32 values mapping to specific positions in 16x8 tile
    # Layout matches tc.py swizzle for output fragment

    output_stores = []

    for m_tile in range(M_TILES):
        for n_tile in range(N_TILES):
            # Calculate base position for this warp's tile in the output
            base_m = warpIdy * WARP_TILE_M + m_tile * TC_M
            base_n = warpIdx * WARP_TILE_N + n_tile * TC_N

            # For m16n8k16 output (16 rows x 8 cols, 4 elements per thread):
            # Each thread's 4 elements map to specific (row, col) positions
            # Pattern derived from WMMA C fragment layout
            for elem in range(4):
                # Row mapping: uses l4,l3,l2 for base row + elem for sub-row
                # 32 threads map to 16 rows: each pair of threads shares rows
                row_idx = (l4 * 8 + l3 * 4 + l2 * 2) + (elem // 2)

                # Col mapping: uses l1,l0 for base col + elem for sub-col
                # 8 columns distributed across 32 threads
                col_idx = l1 * 4 + l0 * 2 + (elem % 2)

                out_row = base_m + row_idx
                out_col = base_n + col_idx

                # Get the accumulated value
                val = c_regs.after(sink)[m_tile, n_tile, elem]
                output_stores.append(c.reshape(BLOCK_M, BLOCK_N)[out_row, out_col].store(val))

    final_sink = UOp.group(*output_stores)

    return final_sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()


def test_matmul_fp16(sink:UOp, N=N):
    """Test function adapted from amd_uop_matmul.py"""
    rng = np.random.default_rng()
    a = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16))
    b = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16))
    hc = Tensor.empty(N, N, dtype=dtypes.float)
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
            err = (hc - tc).square().mean().item()
        print(f"mean squared error {err}")
        if err > 1e-02:
            raise RuntimeError("matmul is wrong!")


if __name__ == "__main__":
    print(f"Building NVIDIA WMMA kernel for {N}x{N} matmul")
    print(f"Configuration:")
    print(f"  Block: {BLOCK_M}x{BLOCK_N}, K-slice: {BLOCK_K}")
    print(f"  Warps: {WARPS_PER_BLOCK}, Warp tile: {WARP_TILE_M}x{WARP_TILE_N}")
    print(f"  Tensor core: m{TC_M}n{TC_N}k{TC_K}")
    print(f"  Tiles per warp: {M_TILES}x{N_TILES} = {M_TILES * N_TILES} WMMA ops per K iteration")

    sink = hand_spec_nv_wmma()

    print("\nKernel built successfully!")
    print("To run: CUDA=1 python extra/gemm/nv_uop_matmul_working.py")

    if Device.DEFAULT == "CUDA":
        test_matmul_fp16(sink, N=N)
