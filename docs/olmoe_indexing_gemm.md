# OLMoE: Why Splitting Indexing from GEMMs Helps

This note explains why the change in `examples/olmoe.py` that separates expert indexing from the
matmul kernels produces a large speedup. It includes the old/new code, the UOps for each case, and
why the fused version creates dynamic access patterns that block GEMM optimizations.

## 1) Previous and new code

Previous (fused gather into GEMM):

```python
# run MoE
x_up_gate = x.dot(self.gate_proj[sel].permute(0, 2, 1)).silu() * \
  x.dot(self.up_proj[sel].permute(0, 2, 1))
x_down = x_up_gate.dot(self.down_proj[sel].permute(0, 2, 1))
```

New (gather first, then GEMM):

```python
# run MoE
selected_gate_projs = self.gate_proj[sel]
selected_up_projs = self.up_proj[sel]
selected_down_projs = self.down_proj[sel]
# Split indexing from dot so dot kernels keep their own optimizations.
selected_gate_projs.realize(selected_up_projs, selected_down_projs)
x_up_gate = x.dot(selected_gate_projs.permute(0, 2, 1)).silu() * \
  x.dot(selected_up_projs.permute(0, 2, 1))
x_down = x_up_gate.dot(selected_down_projs.permute(0, 2, 1))
```

## 2) UOps comparison (fused vs split)

Below are **small, illustrative** UOps from a tiny example (`E=4, K=2, D=4, H=8`). Lines are
wrapped for readability. The key difference is that the *fused* matmul includes indexing logic
(`where`, `reshape`, `expand`) inside the matmul graph, while the *split* matmul is a clean GEMM from
contiguous buffers.

**Fused matmul (gather is inside the GEMM graph):**

```text
c2 = UOp.new_buffer('CPU', 4, dtypes.float, 0)
c10 = UOp.new_buffer('CPU', 2, dtypes.int, 2)
...
c91 = ((c25.reshape((2,1,1,1)).expand((2,4,8,4)) != ...)
  .where(c81.reshape((4,8,4)).reshape((1,4,8,4)).expand((2,4,8,4)), 0.0))
c99 = c2.reshape((1,1,4)).reshape((1,1,1,4)).expand((2,1,8,4)) * \
  c91.r(Ops.ADD, (1,)).reshape((2,8,4)).permute((0, 2, 1))
...
ast = c99.r(Ops.ADD, (3,)).reshape((2,1,8))
```

**Split path: gather as its own kernel:**

```text
c2 = UOp.new_buffer('CPU', 2, dtypes.int, 2)
...
c83 = ((c17.reshape((2,1,1,1)).expand((2,4,8,4)) != ...)
  .where(c73.reshape((4,8,4)).reshape((1,4,8,4)).expand((2,4,8,4)), 0.0))
ast = c83.r(Ops.ADD, (1,)).reshape((2,8,4))
```

**Matmul after realize (clean GEMM input):**

```text
c2 = UOp.new_buffer('CPU', 4, dtypes.float, 0)
c10 = UOp.new_buffer('CPU', 64, dtypes.float, 5)
c17 = c2.reshape((1,1,4)).reshape((1,1,1,4)).expand((2,1,8,4)) * \
  c10.reshape((2,8,4)).permute((0, 2, 1))
ast = c17.r(Ops.ADD, (3,)).reshape((2,1,8))
```

**What changes in practice:**

- **Fused case:** the matmul input `c91` is built by data-dependent indexing (`where`, `expand`).
  The GEMM sees a graph that is *not* a simple contiguous/strided buffer.
- **Split case:** the gather is its own kernel and writes a buffer (`c10`) that the matmul can treat
  as a plain, contiguous input.

## 3) Why the old logic produces dynamic shapes & strides

The logical shapes are still static, but the **memory access pattern is dynamic**:

- `sel` comes from `topk`, which depends on runtime data. The gather `gate_proj[sel]` is not a view
  with fixed strides; it is a **data-dependent gather**.
- In UOps, this appears as `where` + `expand` + `reshape` that synthesizes the gathered tensor
  element-by-element.
- Because the tensor feeding GEMM is **not a contiguous buffer with fixed affine strides**, the
  compiler can't safely apply fast GEMM schedules (tiling, vectorization, tensor-core-like paths).

By realizing the gather first, we materialize a **concrete, contiguous buffer** for the selected
experts. The GEMM then sees a clean input and can use the optimized matmul kernel. That is why the
split approach yields a large speedup even though it adds a small extra kernel.

## Reproducing the UOps

This snippet prints the UOps used above (small shapes for clarity):

```python
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import pyrender

B, T, D = 1, 1, 4
E, H, K = 4, 8, 2

x = Tensor.empty(B, T, D)
gate_proj = Tensor.empty(E, H, D)
sel = Tensor.empty(K, dtype=dtypes.int32)

# fused gather into matmul
fused = x.dot(gate_proj[sel].permute(0, 2, 1))
print(pyrender(fused.uop))

# gather only
selected = gate_proj[sel]
print(pyrender(selected.uop))

# matmul after realize (use a fresh buffer to simulate realized output)
selected_buf = Tensor.empty(K, H, D)
matmul = x.dot(selected_buf.permute(0, 2, 1))
print(pyrender(matmul.uop))
```
