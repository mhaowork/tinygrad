import os
from pathlib import Path

os.environ.setdefault("CACHEDB", str(Path(__file__).resolve().parent.parent / ".cache" / "tinygrad_cache.db"))

import numpy as np
from dataclasses import replace

from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import CompiledRunner
from tinygrad.runtime.ops_cpu import CPUProgram
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN

# Reference assembly from the other Mac - transcribed from screenshot
# Key difference: uses indexed addressing with x8 instead of auto-increment
reduce_asm_ref = """
movi  v0.2d, #0000000000000000
mov   x8, xzr
mov   w9, #0x3ffffc
loop:
add   x10, x1, x8, lsl #4
cmp   x8, x9
ldp   q1, q2, [x10]
fadd  v1.4s, v1.4s, v2.4s
ldp   q2, q3, [x10, #0x20]
add   x8, x8, #4
fadd  v1.4s, v1.4s, v2.4s
fadd  v1.4s, v1.4s, v3.4s
fadd  v0.4s, v0.4s, v1.4s
b.lo  loop
dup   v1.4s, v0.s[1]
dup   v2.4s, v0.s[2]
fadd  v1.4s, v0.4s, v1.4s
dup   v0.4s, v0.s[3]
fadd  v1.4s, v2.4s, v1.4s
fadd  v0.4s, v0.4s, v1.4s
str   s0, [x0]
ret
"""

# Assemble the reference assembly
ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
try:
    arm_bytecode, _ = ks.asm(reduce_asm_ref)
    arm_bytecode = bytes(arm_bytecode)
    print(f"Successfully assembled {len(arm_bytecode)} bytes of ARM64 code")
except Exception as e:
    print(f"Assembly failed: {e}")
    arm_bytecode = None

if __name__ == "__main__":
    np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32) - 0.5
    a = Tensor(np_array).realize()
    with Context(SPLIT_REDUCEOP=0):
        GlobalCounters.reset()
        out = a.sum()
        for i, ei in enumerate(out.schedule()):
            ei.lower()
            if i == 0 and arm_bytecode is not None:
                # Inject the reference assembly
                from tinygrad.device import Device
                prg_spec = ei.prg.p
                prg_spec = replace(prg_spec, name="reduce_ref")
                prg = CompiledRunner(prg_spec)
                # Replace with our hand-assembled code
                dev = Device[prg_spec.device]
                prg._prg = CPUProgram(dev, prg_spec.name, arm_bytecode)
                print(f"Injected reference assembly into kernel")
                ei = replace(ei, prg=prg)
            ei.run()
    np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
    print(out.item())
