#!/usr/bin/env python3
"""
Compare assembly from different implementations
"""
import subprocess
import tempfile
import os

# Our multi-accumulator version (matching what tinygrad generates)
tinygrad_code = """
typedef float float4 __attribute__((aligned(16),vector_size(16)));

void reduce_tinygrad(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};

  int size = 4096 * 4096;
  for (int ridx0 = 0; ridx0 < size; ridx0+=16) {
    acc0 += *((float4*)((data1 + ((ridx0+0) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+1) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+2) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+3) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+4) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+5) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+6) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+7) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+8) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+9) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+10) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+11) << 2))));

    acc0 += *((float4*)((data1 + ((ridx0+12) << 2))));
    acc1 += *((float4*)((data1 + ((ridx0+13) << 2))));
    acc2 += *((float4*)((data1 + ((ridx0+14) << 2))));
    acc3 += *((float4*)((data1 + ((ridx0+15) << 2))));
  }

  float4 total = acc0 + acc1 + acc2 + acc3;
  *(data0 + 0) = total[0] + total[1] + total[2] + total[3];
}
"""

# Optimized version using NEON intrinsics directly (like what Accelerate might do)
neon_code = """
#include <arm_neon.h>

void reduce_neon(float* restrict data0, float* restrict data1) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  int size = 4096 * 4096;
  for (int i = 0; i < size; i += 16) {
    acc0 = vaddq_f32(acc0, vld1q_f32(&data1[i + 0]));
    acc1 = vaddq_f32(acc1, vld1q_f32(&data1[i + 4]));
    acc2 = vaddq_f32(acc2, vld1q_f32(&data1[i + 8]));
    acc3 = vaddq_f32(acc3, vld1q_f32(&data1[i + 12]));
  }

  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc0 = vaddq_f32(acc0, acc2);

  *data0 = vaddvq_f32(acc0);
}
"""

# Even more optimized with unrolling
neon_unrolled_code = """
#include <arm_neon.h>

void reduce_neon_unrolled(float* restrict data0, float* restrict data1) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  float32x4_t acc4 = vdupq_n_f32(0.0f);
  float32x4_t acc5 = vdupq_n_f32(0.0f);
  float32x4_t acc6 = vdupq_n_f32(0.0f);
  float32x4_t acc7 = vdupq_n_f32(0.0f);

  int size = 4096 * 4096;
  for (int i = 0; i < size; i += 32) {
    acc0 = vaddq_f32(acc0, vld1q_f32(&data1[i + 0]));
    acc1 = vaddq_f32(acc1, vld1q_f32(&data1[i + 4]));
    acc2 = vaddq_f32(acc2, vld1q_f32(&data1[i + 8]));
    acc3 = vaddq_f32(acc3, vld1q_f32(&data1[i + 12]));
    acc4 = vaddq_f32(acc4, vld1q_f32(&data1[i + 16]));
    acc5 = vaddq_f32(acc5, vld1q_f32(&data1[i + 20]));
    acc6 = vaddq_f32(acc6, vld1q_f32(&data1[i + 24]));
    acc7 = vaddq_f32(acc7, vld1q_f32(&data1[i + 28]));
  }

  acc0 = vaddq_f32(acc0, acc1);
  acc2 = vaddq_f32(acc2, acc3);
  acc4 = vaddq_f32(acc4, acc5);
  acc6 = vaddq_f32(acc6, acc7);
  acc0 = vaddq_f32(acc0, acc2);
  acc4 = vaddq_f32(acc4, acc6);
  acc0 = vaddq_f32(acc0, acc4);

  *data0 = vaddvq_f32(acc0);
}
"""

def compile_and_disassemble(code, func_name, opt_level="-O3"):
    """Compile C code and disassemble the specified function"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        c_file = f.name

    try:
        # Compile to object file
        o_file = c_file.replace('.c', '.o')
        compile_cmd = ['clang', opt_level, '-march=native', '-c', c_file, '-o', o_file]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return None

        # Disassemble - on macOS use otool
        result = subprocess.run(
            ['otool', '-tv', o_file],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            # Try with llvm-objdump
            result = subprocess.run(
                ['llvm-objdump', '-d', o_file],
                capture_output=True, text=True
            )

        # Extract just the function we care about
        lines = result.stdout.split('\n')
        in_func = False
        asm_lines = []

        for line in lines:
            if func_name in line and '<' in line:
                in_func = True
                asm_lines.append(line)
            elif in_func:
                if line.strip() and not line.startswith('_'):
                    asm_lines.append(line)
                elif line.strip().startswith('_') or (line.strip() and ':' in line and '<' in line):
                    # Hit next function
                    break

        os.unlink(o_file)
        return '\n'.join(asm_lines)

    finally:
        os.unlink(c_file)

print("=" * 80)
print("COMPARING ASSEMBLY GENERATION FOR REDUCTION")
print("=" * 80)

print("\n### 1. Tinygrad-style (4 acc, vector syntax)")
print("-" * 80)
asm = compile_and_disassemble(tinygrad_code, "reduce_tinygrad")
if asm:
    # Count instructions in main loop
    lines = [l for l in asm.split('\n') if l.strip() and not l.startswith('<')]
    print(f"Total instructions: {len(lines)}")
    print("\nFirst 40 lines:")
    for line in lines[:40]:
        print(line)
else:
    print("Failed to disassemble")

print("\n### 2. NEON intrinsics (4 acc)")
print("-" * 80)
asm = compile_and_disassemble(neon_code, "reduce_neon")
if asm:
    lines = [l for l in asm.split('\n') if l.strip() and not l.startswith('<')]
    print(f"Total instructions: {len(lines)}")
    print("\nFirst 40 lines:")
    for line in lines[:40]:
        print(line)
else:
    print("Failed to disassemble")

print("\n### 3. NEON intrinsics (8 acc, more unrolling)")
print("-" * 80)
asm = compile_and_disassemble(neon_unrolled_code, "reduce_neon_unrolled")
if asm:
    lines = [l for l in asm.split('\n') if l.strip() and not l.startswith('<')]
    print(f"Total instructions: {len(lines)}")
    print("\nFirst 40 lines:")
    for line in lines[:40]:
        print(line)
else:
    print("Failed to disassemble")

print("\n" + "=" * 80)
print("KEY DIFFERENCES TO LOOK FOR:")
print("=" * 80)
print("1. Number of vector registers used (more = better ILP)")
print("2. Load patterns (ldp vs individual ldr)")
print("3. Dependency chains (consecutive fadds on same register = bad)")
print("4. Loop overhead (branches, counter updates)")
print("5. Use of specialized instructions (faddp for horizontal add)")
