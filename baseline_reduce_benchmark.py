import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from tinygrad import Tensor, GlobalCounters, Context

def benchmark_reduce(warmup_iters=5, test_iters=10):
    """Benchmark tinygrad's auto-generated reduce (sum) performance"""

    # Create test data
    np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5
    a = Tensor(np_array).realize()

    print("=" * 70)
    print("Tinygrad Auto-Generated Reduce (Sum) - Baseline Benchmark")
    print("=" * 70)
    print(f"Array size: 4096 x 4096 = {4096*4096:,} floats ({4096*4096*4/1e6:.1f} MB)")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Test iterations: {test_iters}")
    print("=" * 70)
    print("Note: Set DEBUG=7 CPU=1 CPU_CLANG=1 to see per-iteration timing")
    print("=" * 70)

    with Context(SPLIT_REDUCEOP=0):
        # Warmup phase
        print("\n[Warmup Phase]")
        for i in range(warmup_iters):
            GlobalCounters.reset()
            out = a.sum()
            out.realize()

        # Test phase
        print(f"\n[Test Phase - {test_iters} iterations]")
        for i in range(test_iters):
            GlobalCounters.reset()
            out = a.sum()
            out.realize()

        result = out.item()

    print("\n" + "=" * 70)
    print(f"Result: {result:.2f} (expected: {np_array.sum():.2f})")

    # Verify correctness
    np.testing.assert_allclose(result, np_array.sum(), atol=1, rtol=1e-4)
    print("✓ Correctness verified")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_reduce()
