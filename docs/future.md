1. The Core Philosophy: "The Unstoppable Stream"

The system is divided into Planning (Host) and Execution (Device). Once Execution begins, all logical branching, kernel selection, and memory offsets are immutable. The GPU never waits for a Host decision.
2. Phase 1: Lazy Graph Construction & Backend Propagation

    Lazy Tensors: Tensor operations do not trigger compute. They append nodes to a Device-Agnostic DAG.

    Flag Propagation: As nodes are added, a "Hardware Preference" flag propagates.

        Rule: If an operation requires a specific backend (e.g., a CUDA-only custom op), all connected nodes default to that backend to minimize expensive Host-to-Device (H2D) copies.

3. Phase 2: IR Transformation & Aggressive Optimization

The DAG is lowered into a Linear IR (Intermediate Representation).

    Optimization Passes:

        Dead Node Elimination: Remove tensors that don't contribute to the final output or gradient.

        Vertical Fusion: Combine Mul + Add into a single Fused_Mul_Add IR node.

        Horizontal Fusion: Group independent element-wise ops into a single kernel launch to maximize GPU occupancy.

    Output: An optimized IR Sequence where each node represents a specific "Kernel Specification."

4. Phase 3: The "MakeKernel" Factory (Demand-Driven Codegen)

This is the bridge between IR and raw CUDA.

    The Template Library: A collection of highly parameterized .cu templates (using C++ templates or string injectors) for operations like Map, Reduce, Broadcast, and GEMM.

    Pattern Matching:

        System takes the IR Spec (Shape, Stride, Dtype, Op-Chain).

        Searches the Kernel Hashtable (Key: IR Hash).

        On Miss: Finds the closest matching Template → Injects specific constants (Strides/Shapes) → Generates specialized CUDA code → Compiles via NVRTC → Stores in Hashtable.

        On Hit: Retrieves the function pointer immediately.

5. Phase 4: Execution & Result Memoization (JIT Cache)

The "Execution Cycle" is a tight loop through the optimized IR.

    AOT Dispatch: The engine falls through a pre-calculated switch/lookup table of the kernels generated in Phase 3.

    JIT Result Cache: * Key: Hash(DAG_Structure + Input_Metadata).

        Value: Device_Pointer to the computed result.

        Note: If the path is cached, the kernel launch is skipped entirely.

6. Phase 5: Autograd Integration (The r_OP Link)

To ensure the backward pass is as optimized as the forward pass:

    Root Op Mapping: Every output tensor stores a reference (r_OP) to the specific kernel that birthed it.

    Gradient Tape: During the backward walk, the system looks at the r_OP to identify the exact specialized kernel needed for the derivative, ensuring symmetry in optimization.
