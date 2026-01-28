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

def benchmark_vdsp(a):
    """Benchmark raw vDSP performance"""
    if not has_vdsp:
        return None, None

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
    start = time.perf_counter()
    for _ in range(100):
        vDSP_sve(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.size
        )
    end = time.perf_counter()

    avg_time = (end - start) / 100 * 1000  # ms
    return avg_time, result[0]

def benchmark_torch(a):
    """Benchmark PyTorch (which also uses Accelerate on macOS)"""
    import torch
    a_torch = torch.from_numpy(a)

    # Warmup
    for _ in range(10):
        result = a_torch.sum().item()  # Force eval with .item()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        result = a_torch.sum().item()  # Force eval
    end = time.perf_counter()

    avg_time = (end - start) / 100 * 1000  # ms
    return avg_time, result

def benchmark_numpy(a):
    """Benchmark NumPy (baseline)"""
    # Warmup
    for _ in range(10):
        result = a.sum()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        result = a.sum()
    end = time.perf_counter()

    avg_time = (end - start) / 100 * 1000  # ms
    return avg_time, result

def benchmark_tinygrad(a):
    """Benchmark tinygrad"""
    from tinygrad import Tensor
    a_tg = Tensor(a).realize()

    # Warmup
    for _ in range(10):
        result = a_tg.sum().numpy()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        result = a_tg.sum().numpy()
    end = time.perf_counter()

    avg_time = (end - start) / 100 * 1000  # ms
    return avg_time, result

if __name__ == "__main__":
    for size in [2048, 4096]:
        print(f"\n{'='*70}")
        print(f"Testing {size}x{size} ({size*size} elements, {size*size*4/1024/1024:.1f} MB)")
        print(f"{'='*70}")

        a = np.random.random((size, size)).astype(np.float32)
        expected = a.sum()

        results = []

        # Test vDSP
        if has_vdsp:
            vdsp_time, vdsp_result = benchmark_vdsp(a)
            results.append(("vDSP (raw)", vdsp_time, vdsp_result))
            print(f"vDSP (raw):     {vdsp_time:6.2f} ms  result={vdsp_result:.2f}")

        # Test NumPy
        numpy_time, numpy_result = benchmark_numpy(a)
        results.append(("NumPy", numpy_time, numpy_result))
        print(f"NumPy:          {numpy_time:6.2f} ms  result={numpy_result:.2f}")

        # Test PyTorch
        torch_time, torch_result = benchmark_torch(a)
        results.append(("PyTorch", torch_time, torch_result))
        print(f"PyTorch:        {torch_time:6.2f} ms  result={torch_result:.2f}")

        # Test tinygrad
        tg_time, tg_result = benchmark_tinygrad(a)
        results.append(("tinygrad", tg_time, tg_result))
        print(f"tinygrad:       {tg_time:6.2f} ms  result={tg_result:.2f}")

        # Comparison
        print(f"\n{'Speedup vs vDSP':^70}")
        print(f"{'-'*70}")
        if has_vdsp:
            for name, t, _ in results:
                if name != "vDSP (raw)":
                    speedup = vdsp_time / t
                    print(f"{name:15s}: {speedup:5.2f}x {'faster' if speedup > 1 else 'slower'}")
