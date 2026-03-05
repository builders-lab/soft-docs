
# Custom Tensor Runtime
### Systems-Level Deep Learning Research

**Project:** Cost–Benefit Analysis of CPU vs GPU Computation in Deep Learning  
**Institution:** GLA University, Mathura  
**Program:** B.Tech – Computer Science & Engineering  
**Academic Year:** 2025–2026

---

# 1. Overview

Modern deep learning frameworks such as PyTorch and TensorFlow provide powerful abstractions for building neural networks. However, these abstractions often hide the underlying computational behavior of the hardware.

For small-to-medium workloads, overhead introduced by Python runtimes, automatic memory management, and large software stacks can obscure the true performance characteristics of CPU and GPU computation.

This project explores a different approach.

The goal is to build a **minimal, transparent tensor runtime written in C++**, designed specifically for studying the interaction between neural network computation and hardware execution.

Instead of prioritizing convenience or ecosystem size, the runtime focuses on:

- explicit memory control  
- predictable execution behavior  
- direct interaction with CPU and GPU computation  

By implementing custom CUDA kernels and manual memory management without relying on heavy libraries such as cuDNN, the project provides a **white-box environment for studying hardware efficiency in deep learning workloads**.

---

# 2. Research Motivation

The project investigates a fundamental systems question:

> When does GPU acceleration actually become beneficial compared to CPU execution?

While GPUs excel at large-scale parallel computation, data transfer overhead and kernel launch latency can make CPU execution competitive for smaller workloads.

This runtime allows controlled experiments to explore questions such as:

- At what tensor size does `cudaMemcpy` overhead outweigh GPU parallelism gains?
- Can a dynamic execution scheduler improve performance by running early network layers on the CPU and wider layers on the GPU?
- How does manual memory management compare to managed runtimes in terms of memory usage and latency?

The long-term objective is to identify the **precise crossover point where GPU acceleration becomes advantageous for neural network workloads**.

---

# 3. Engineering Philosophy

The runtime is built on a strict design principle:

**Clarity > Flexibility**

Unlike production frameworks designed for broad compatibility, this system intentionally restricts certain behaviors in order to maintain architectural transparency.

Key design decisions include:

### Opaque Tensor Architecture

Tensor objects are exposed through stable interfaces such as:

- `tensor_t`
- `tensor_graph_t`
- `tensor_pool_t`

The internal memory layout remains hidden, allowing the backend implementation to evolve without breaking the API.

---

### Explicit Output Operations

All tensor operations require **pre-allocated output buffers**.

Example design pattern:
    - tensor_matmul(out, a, b)

This avoids hidden allocations and improves memory predictability.

---

### Strict Device Coherence

Tensor operations enforce device consistency.

Operations will **fault if tensors from different devices are mixed implicitly**.  
This avoids silent CPU↔GPU transfers and ensures predictable performance measurements.

---

# 4. Technical Environment

The runtime is developed using the following environment:

| Component | Specification |
|----------|---------------|
| Language | C++17 or later |
| GPU Compute | NVIDIA CUDA Toolkit (11.0+) |
| GPU Hardware | NVIDIA GPU (Compute Capability ≥ 6.0) |
| CPU | x86-64 architecture |
| Build System | CMake |
| Operating System | Linux |

The project intentionally avoids heavy deep-learning libraries to maintain full control over memory management and kernel execution.

---

# 5. Experimental Validation

To validate correctness and measure performance, the system will be evaluated using the **MNIST handwritten digit dataset**.

MNIST provides a controlled workload that allows:

- rapid training iteration
- predictable convergence behavior
- minimal disk I/O interference

The dataset can be fully loaded into RAM or VRAM, ensuring that benchmarking results reflect **compute performance rather than storage latency**.

Initial experiments will train networks up to **10 layers deep** to confirm functional correctness and runtime stability.

---

# 6. Project Scope

This runtime is not intended to replace existing frameworks. Instead, it serves as:

- a **research platform for studying ML system performance**
- a **teaching tool for understanding tensor runtimes**
- a **systems experiment in hardware-aware neural network execution**

By reducing abstraction layers, the project aims to provide deeper insight into how modern machine learning workloads interact with real hardware.

---

# Future Documentation

Additional documentation sections include:

- Architecture Overview  
- Tensor Runtime API  
- Device Scheduling Model  
- CUDA Kernel Implementation  
- Benchmark Results  
