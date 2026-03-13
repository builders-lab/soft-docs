# Things to build for soft-cuda

## First tensor system

N-dimensional tensor struct

GPU memory allocation (cudaMalloc / cudaFree)

Host ↔ Device transfers (cudaMemcpy)

Stride + shape metadata


something like this 

```C
struct Tensor {
    float* data;      
    int* shape;
    int* strides;
    int dims;
};
```

## Core Linear Algebra Kernels

Matrix multiplication (GEMM)

Vector add

Elementwise multiply

Bias addition

Reduction (sum, mean)

Transpose (for backprop)

Softmax

ReLU (and its derivative)


## Automatic Differentiation Engine


Computational graph node

Each op stores:

- Forward output

- Backward function pointer

Reverse-mode backprop traversal

```C
struct Node {
    Tensor* value;
    Tensor* grad;
    void (*backward)(struct Node*);
    Node** parents;
};
```

## Optimizer

SGD
Adam
```math
param -= lr * grad
```

## Loss Functions

Minimum:

- MSE

- Cross Entropy

Cross entropy requires:

- Stable softmax (log-sum-exp trick)

## Memory Strategy

- Preallocate memory pools

- Avoid excessive cudaMalloc in training loop

- Consider unified memory only if you enjoy random slowdowns

Forward activation storage

Gradient storage

Parameter storage


---

## Slice of life

CUDA streams

Kernel fusion (activation + bias add)

Mixed precision (FP16)

cuDNN for convolution

CUBLAS Lt for better GEMM

---

## How to prepare

### CUDA Fundamentals

Thread hierarchy (grid, block, warp)

Shared memory

Memory coalescing

Bank conflicts

Occupancy

Official:

- NVIDIA CUDA Programming Guide

- CUDA Samples on GitHub


### Matrix Multiplication Optimization
“Anatomy of High Performance Matrix Multiplication”

NVIDIA blog on tiled GEMM

Implement:

- Naive GEMM

- Tiled shared memory GEMM

- Compare performance


## Backprop From Scratch

Before CUDA:

- Implement 10-layer NN in pure C or Python (no autograd)

- Then build your own reverse-mode engine

## Study These Frameworks

Minimal frameworks to read:

- tinygrad

- micrograd

- Darknet

- OneFlow (for large scale view)

## Rought system architecture

GPU Memory Manager
↓
Tensor Abstraction
↓
Core Math Kernels (GEMM + elementwise)
↓
Autograd Engine
↓
Loss Functions
↓
Optimizer
↓
Model API


