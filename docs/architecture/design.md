# soft-cuda Architecture Design Document

!!! info "Document Status"
    **Frozen Design Decisions — March 2026**  
    Author: Aakarsh | License: BSD-2-Clause | [github.com/builders-lab/soft-cuda](https://github.com/builders-lab/soft-cuda)

---

## 1. Project Overview

soft-cuda is a CUDA/C++ tensor library for research, built on lazy evaluation, a bump allocator memory pool, and a computation graph with ahead-of-time backend assignment. It is not a production framework — it is a research instrument designed for transparency, reproducibility, and first-principles understanding.

!!! quote "Core Thesis"
    Backend selection decisions should be made once, explicitly, and frozen into the computation graph before execution — not dispatched at runtime on every operation.

---

## 2. Core Data Structure: `tensor_instance`

The fundamental unit of computation. Every tensor, operation result, and gradient is a `tensor_instance`.

| Field | Description |
|---|---|
| `ndims` | Number of dimensions |
| `dims[]` | Size along each dimension |
| `stride[]` | Stride for each dimension (row-major) |
| `void *data` | Raw data pointer (CPU or GPU depending on device) |
| `op` | Operation that produced this tensor (`ADD`, `MUL`, `RELU`, `NONE`) |
| `a, b` | Pointers to input tensors in the computation graph |
| `device` | Which backend owns this tensor's data (`CPU` / `CUDA`) |
| `grad_compute` | Function pointer for gradient computation |
| `grad` | Pointer to gradient `tensor_instance` |
| `evaluated` | Memoization bit — set after DFS evaluation, prevents re-evaluation |
| `backend_id` | Baked-in backend assignment from `CONFIG.soft` at graph compile time |

---

## 3. Memory Pool: Bump Allocator

A bump allocator pool manages all tensor memory. It is fast (O(1) allocation), avoids malloc fragmentation, and maps cleanly to both CPU RAM and CUDA VRAM.

### 3.1 Pool Architecture

| Parameter | Behaviour |
|---|---|
| `device_type` | CPU allocates via `malloc`; CUDA allocates via `cudaMalloc` |
| `block_size` | Configurable. Default defined in `DEFAULT.soft` |
| bump pointer | Advances on every allocation — no free per-tensor |
| pool reset | Entire pool released at end of computation graph execution |

### 3.2 VRAM Extension

When `device_type = CUDA`, the pool calls `cudaMalloc` for the backing block and `cudaMemcpy` for data transfers. The allocator interface is identical from the caller's perspective — device type is an internal concern of the pool.

---

## 4. Computation Graph & Lazy Evaluation

Operations do not execute immediately. Instead, they build a directed acyclic graph (DAG) of `tensor_instance` nodes. Execution is deferred until explicitly triggered, at which point the graph is compiled and then evaluated.

### 4.1 Lazy Evaluation Flow

| Phase | What Happens |
|---|---|
| Define ops | `tensor_instance` nodes created, `op/a/b` fields populated. No computation. |
| Build graph | DAG is fully constructed in memory. |
| Compile graph | `CONFIG.soft` consulted. Each node gets `backend_id` baked in based on `(op, size_bucket)`. |
| DFS Evaluate | Post-order DFS traversal. Each node fires its pre-assigned backend. `evaluated` bit set for memoization. |

### 4.2 DFS Evaluation

Post-order DFS ensures inputs are always evaluated before the operation that consumes them. The `evaluated` bit prevents any node from being computed twice in a shared-subexpression graph.

---

## 5. `CONFIG.soft` — The Device Profile

`CONFIG.soft` is the frozen device profile. It is generated once by `soft_init` and consulted at graph compile time to assign backends to graph nodes. It is human-readable and version-controlled alongside research experiments.

### 5.1 File Schema

```ini
[meta]
soft_version = 0.1.0
device_hash  = <sha256 of gpu_name+compute_cap+vram+driver>
generated_at = 2026-03-18T09:00:00Z

[device]
type               = cuda
compute_capability = 8.6
vram_mb            = 8192

[ops]
# format: op_name | size_bucket -> backend
matmul | size < 128        = cpu_blas
matmul | 128 <= size < 512 = cuda_naive
matmul | size >= 512       = cuda_tiled
relu   | any               = cuda_elementwise
add    | any               = cuda_elementwise

[pool]
device     = cuda
block_size = 2097152
```

### 5.2 Config Invalidation

On library load, `device_hash` is recomputed and compared against `CONFIG.soft`. If the hash mismatches, a warning is printed and the researcher is prompted to re-run `soft_init`. No silent stale configs.

### 5.3 Default Config

`DEFAULT.soft` ships with the library. It assigns all ops to `cpu_blas` / `cpu_fallback`. Used when `soft_init` has not been run. Ensures the library works on a fresh clone with no CUDA setup required.

---

## 6. `soft_init` — The Explicit Init Process

`soft_init` is a separate, visible process. It is not called automatically. The researcher runs it once per machine/device and sees exactly what decisions are being made.

### 6.1 What `soft_init` Does

1. Detects available hardware (GPU name, compute capability, VRAM, driver version)
2. Computes `device_hash`
3. Runs a benchmark sweep: for each registered `op` × each `size_bucket`, measures wall time on CPU and CUDA
4. Selects the winner per `(op, size_bucket)` pair
5. Writes `CONFIG.soft` with full results visible to the researcher

### 6.2 Design Philosophy

No hidden calibration. The researcher sees the benchmark output, the device hash, and the config being written. This is a deliberate transparency decision appropriate for a research library where reproducibility demands knowing exactly what backend decision was made and why.

!!! note "Key Philosophical Difference"
    Production frameworks like cuDNN and TVM hide backend selection entirely. `soft_init` makes every decision visible and version-controllable.

---

## 7. Backend Architecture

Backends are registered computation providers. Each backend implements the same op interface. At graph compile time, nodes are assigned a `backend_id`. At evaluation time, the DFS executor calls the pre-assigned backend — zero runtime dispatch overhead.

| Backend ID | Description |
|---|---|
| `cpu_fallback` | Pure C naive implementation. Always available. Reference correctness. |
| `cpu_blas` | libopenblas. Used for matmul at small-to-medium sizes. |
| `cuda_naive` | Basic CUDA kernel. No tiling. Used for medium sizes. |
| `cuda_tiled` | Tiled CUDA kernel with shared memory. Used for large matmul. |
| `cuda_elementwise` | Simple parallel CUDA kernel. Used for relu, add, etc. |

### 7.1 The Key Insight: Compile-Time Baking

Because soft-cuda uses lazy evaluation, the backend decision does not happen at op-definition time. It happens at graph compilation time — after the full graph is built but before any computation runs. This means:

- The DFS executor never queries `CONFIG.soft` — backends are already baked into nodes
- Adjacent CUDA-assigned nodes can potentially be fused into a single kernel launch *(future work)*
- Graph-level optimization becomes possible because assignment sees the whole graph before execution begins

---

## 8. April 2026 Research Target

!!! question "Research Question"
    At what tensor size does GPU PCI-e overhead get overcome by GPU parallelism?

`soft_init`'s benchmark sweep IS the experiment. `CONFIG.soft` IS the result. The threshold curve IS the finding.

| Milestone | Target Date |
|---|---|
| CUDA backend for matmul, relu, add (naive kernels) | April 1 |
| `soft_init` benchmark sweep across size buckets | April 7 |
| `CONFIG.soft` generation with threshold detection | April 10 |
| Research writeup from `CONFIG.soft` data | April 13 |
| April 15 submission | April 15 |

---

## 9. Full Framework Roadmap (Post April)

| Phase | Work |
|---|---|
| **Now (inject)** | `CONFIG.soft` schema + parser, `soft_init` stub, `backend_id` field on graph nodes |
| **Autograd** | Build on graph nodes that already carry `backend_id` — no philosophy mismatch |
| **Graph evaluator** | DFS reads `backend_id` field, dispatches pre-assigned backend |
| **CUDA integration** | Register CUDA backends — architecture already knows what a backend is |
| **Graph fusion** | Identify adjacent CUDA nodes, merge kernel launches |
| **Full CONFIG.soft** | Benchmark sweep covers autograd ops, full `(op × size_bucket)` table |

!!! warning "Inject Now"
    Inject `CONFIG.soft` and `backend_id` before building autograd. Refactoring a design philosophy into a half-built system is harder than building on the right foundation from the start.

---

## 10. Frozen Design Decisions

These decisions are settled. They should not be revisited without strong evidence.

| Decision | Rationale |
|---|---|
| BSD-2-Clause license | Maximum freedom. No copyleft constraints. Research-first. |
| Explicit `soft_init`, no auto-detection | Research library. Transparency and reproducibility over convenience. |
| `CONFIG.soft` is human-readable | Researcher must be able to inspect and version-control device decisions. |
| Device hash for invalidation | Stale configs on new hardware must be caught explicitly, never silently. |
| `(op × size_bucket)` granularity | op-only granularity is too coarse. A 64×64 and 4096×4096 matmul may want different backends. |
| Backend baked at graph compile time | Zero dispatch overhead at execution. Enables future graph-level fusion. |
| `DEFAULT.soft` ships with library | Fresh clone must work. CPU fallback is always safe. |
| Bump allocator, not malloc per tensor | O(1) allocation, no fragmentation, maps cleanly to `cudaMalloc`. |
| `evaluated` bit for memoization | Shared subexpressions in DAG must not be recomputed. |
| No LSP, no code gen from AI | Maximum friction development. Genuine understanding over speed. |

---

*soft-cuda — Design frozen March 2026*  
*BSD-2-Clause | [github.com/builders-lab/soft-cuda](https://github.com/builders-lab/soft-cuda)*
