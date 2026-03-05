# Tensor Runtime (Soft) – Architecture Decisions (Current)

## 1. Core Philosophy
The library is designed as a research-oriented tensor runtime, prioritizing:
* Performance
* Explicit control
* Predictable execution

It is **not** a convenience framework. 

**Key principle:** `clarity > flexibility`
Because of this, some “nice” behaviors (such as implicit device copies) are intentionally disallowed.

---

## 2. Opaque Tensor Architecture
The public API exposes opaque types. 

**Core Concepts:**
* `tensor_t`
* `tensor_pool_t`
* `tensor_graph_t`

Users cannot access the internal structure. 

**Benefits:**
* Stable public API
* Internal layout can evolve freely
* Backend implementations can change without breaking user code

> **Note:** This is a common pattern used in systems like the PyTorch tensor abstraction and TensorFlow tensor handles.

---

## 3. Memory Model – Tensor Pool
Tensors are allocated from a memory pool.

**Concept Flow:**
`tensor_pool` → `tensor allocations`

**Initial Implementation:** Bump allocator
* Fast allocation
* No fragmentation
* Simple implementation

*The free operation is currently a placeholder and not yet implemented.*

---

## 4. Explicit Output Operations
Operations follow a pre-allocated output pattern.

**Example Concept:**
`tensor_op(out, a, b)` *(Instead of: `return new_tensor`)*

**Advantages:**
* Avoids heap allocations
* Reuses memory
* Predictable performance

> **Note:** Inspired by HPC libraries such as BLAS and cuBLAS.

---

## 5. Tensor Device Awareness
Each tensor knows the device it resides on (CPU or GPU).

**Future Goal:** The tensor runtime decides the optimal execution device based on:
* Operation cost
* Tensor size
* Hardware latency

*A dynamic scheduling system is planned but currently postponed.*

---

## 6. Device Mismatch Policy
**Strict device coherence rule:** `device(x) == device(y) == device(out)`

If a mismatch occurs, it results in a **runtime fault**. There is no implicit data transfer.

**Reasoning:**
* Predictable performance
* Simpler runtime
* Research clarity

---

## 7. Hardware Profiling Runtime
The library includes a small runtime profiler.

**First Execution Flow:**
1. Run microbenchmarks
2. Measure CPU/GPU latency
3. Generate `CONFIG.soft`

The `CONFIG.soft` file stores hardware characteristics, operation performance data, and device selection heuristics.

**Runtime Behavior:**
* **If** `CONFIG.soft` exists → load profile
* **Else** → run profiler

---

## 8. Offline Autotuning
The profiling system determines optimal execution based on tensor size, operation type, and hardware characteristics.

**Example Concept:**
* Small ops → CPU
* Large ops → GPU

This information is securely stored in `CONFIG.soft`.

---

## 9. Future Hardware Profile Database
**Possible Future Improvement:** Embed known hardware profiles.

**Concept:** `device signature → configuration`
*(Example: GPU model + CPU model → scheduling config)*

This allows the system to skip profiling on known hardware.

---

## 10. Tensor Shape Representation
**Current Design:** `uint32_t* dims`

The shape is stored as a sentinel-terminated array.
* **Example:** `[3, 4, 5, 0]`
* **Max dimensions defined as:** `TENSOR_MAX_DIMS = 8`

---

## 11. Autograd System (Planned)
The autograd system is not yet stabilized.

**Planned Design:**
* Tensor operations build a computation graph.
* Loss tensor triggers the backward pass.

**Backward Entry Point:** `tensor_backward(loss_tensor)`
**Graph Structure Representation:** `tensor_graph_t`

---

## 12. Optimizer Interface
**Initial Optimizer Template:** SGD

**Example Concept:** `tensor_sgd_template(parameters, learning_rate)`

Future optimizers are expected to seamlessly plug into this same structure.

---

## 13. Activation and Loss Operations
Initial built-in operations form a minimal core set for neural network experimentation:
* Matrix multiplication
* Transpose
* Addition
* Scalar multiplication
* ReLU activation
* MSE loss

---

## 14. Project Organization Decisions
Team responsibilities have been clearly separated for this operation.

**Subsystem Ownership:**

* **Mathematical Engine:** Aakarsh + Zoya

* **Neural/Data Pipeline:** Vishal

* **QA / Testing / CI:** Anmol

* **CPython API Integration:** Aadya *(pending)*

*Frontend and documentation operations are postponed until the backend architecture stabilizes.*

---

## 15. Feature Deferral
The dynamic compute switching system has been postponed.

**Reason:** To avoid premature optimization and maintain focus on securing the core tensor engine first.

---

## Overall Architecture Vision
The system aims to become:
`Small tensor runtime` + `Explicit memory control` + `Hardware-aware execution` + `Basic autograd`

**Mission Objective:** A minimal deep-learning execution engine, designed strictly for experimentation rather than end-user convenience.
