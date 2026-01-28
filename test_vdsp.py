#!/usr/bin/env python3
"""Quick test to see vDSP performance vs PyTorch vs tinygrad"""

import ctypes
import numpy as np
import time

# Load Accelerate framework
try:
    accelerate = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')

    # vDSP_sve: Vector sum, single-precision
    # void vDSP_sve(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N)
    vDSP_sve = accelerate.vDSP_sve
    vDSP_sve.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # input array
        ctypes.c_long,                    # stride
        ctypes.POINTER(ctypes.c_float),  # output scalar
        ctypes.c_ulong                    # length
    ]
    vDSP_sve.restype = None
    has_vdsp = True
    print("✅ vDSP loaded successfully")
except Exception as e:
    print(f"❌ Failed to load vDSP: {e}")
    has_vdsp = False

def benchmark_vdsp(size):
    """Benchmark raw vDSP performance"""
    if not has_vdsp:
        return None

    a = np.random.random((size, size)).astype(np.float32)
    result = np.zeros(1, dtype=np.float32)

    # Warmup
    for _ in range(10):
        vDSP_sve(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,  # stride
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.size
        )

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        vDSP_sve(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.size
        )
        times.append(time.perf_counter() - start)

    avg_time = np.median(times) * 1000  # ms
    return avg_time, result[0], a

def benchmark_torch(a):
    """Benchmark PyTorch (which also uses Accelerate on macOS)"""
    import torch
    a_torch = torch.from_numpy(a)

    # Warmup
    for _ in range(10):
        result = a_torch.sum()

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = a_torch.sum()
        times.append(time.perf_counter() - start)

    avg_time = np.median(times) * 1000  # ms
    return avg_time, result.item()

def benchmark_tinygrad(a):
    """Benchmark tinygrad"""
    from tinygrad import Tensor
    a_tg = Tensor(a).realize()

    # Warmup
    for _ in range(10):
        result = a_tg.sum().realize()

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = a_tg.sum().realize()
        times.append(time.perf_counter() - start)

    avg_time = np.median(times) * 1000  # ms
    return avg_time, result.numpy()

if __name__ == "__main__":
    for size in [2048, 4096]:
        print(f"\n{'='*60}")
        print(f"Testing {size}x{size} ({size*size} elements)")
        print(f"{'='*60}")

        # Test vDSP
        if has_vdsp:
            vdsp_time, vdsp_result, a = benchmark_vdsp(size)
            print(f"vDSP (raw):     {vdsp_time:.2f} ms  result={vdsp_result:.2f}")
        else:
            print("vDSP not available")
            a = np.random.random((size, size)).astype(np.float32)
            vdsp_time = None

        # Test PyTorch
        torch_time, torch_result = benchmark_torch(a)
        print(f"PyTorch:        {torch_time:.2f} ms  result={torch_result:.2f}")

        # Test tinygrad
        tg_time, tg_result = benchmark_tinygrad(a)
        print(f"tinygrad:       {tg_time:.2f} ms  result={tg_result:.2f}")

        # Comparison
        print(f"\n{'Comparison':^60}")
        print(f"{'-'*60}")
        if vdsp_time:
            print(f"PyTorch vs vDSP:    {torch_time/vdsp_time:.2f}x")
            print(f"tinygrad vs vDSP:   {tg_time/vdsp_time:.2f}x")
        print(f"tinygrad vs PyTorch: {tg_time/torch_time:.2f}x")
