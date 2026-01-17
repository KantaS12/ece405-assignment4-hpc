# Glossary of Terms: CS336 Assignment 2 (Systems/HPC)

This glossary contains key terms and definitions from the Systems and Parallelism handout, arranged chronologically as they appear in the document.

## 1. Profiling and Benchmarking
*   **Asynchronous CUDA Calls**: GPU operations (like `torch.matmul`) that return control to the CPU immediately before the computation finishes.
*   **torch.cuda.synchronize()**: A PyTorch function that blocks CPU execution until all preceding GPU kernels have completed, essential for accurate timing.
*   **Warm-up Steps**: Initial execution iterations performed before benchmarking to ensure kernels are JIT-compiled and GPU caches are initialized.
*   **Execution Profiler**: A tool (like `cProfile` or `nsys`) that instruments code to provide detailed statistics on function call counts and runtimes.
*   **NVIDIA Nsight Systems (nsys)**: A system-wide profiling tool used to visualize CPU and GPU activity, including CUDA kernels and memory transfers.
*   **NVTX Ranges**: Annotations used to mark specific regions of code for visualization in the Nsight Systems timeline.

## 2. Mixed Precision and Memory
*   **Mixed Precision**: A training technique where certain operations use lower precision (FP16/BF16) to leverage Tensor Cores, while critical accumulations remain in FP32.
*   **Tensor Cores**: Specialized GPU hardware designed to accelerate matrix multiplications at reduced precision (FP16, BF16, TF32).
*   **Loss Scaling**: A technique used with FP16 to prevent small gradient values from underflowing to zero by scaling the loss up before backprop.
*   **Bfloat16 (BF16)**: A 16-bit brain floating-point format with the same dynamic range as FP32 but lower precision, offering more stability than FP16.
*   **torch.autocast**: A PyTorch context manager that automatically handles the casting of operations to lower precision during the forward pass.
*   **PyTorch Memory Profiler**: A tool for tracking GPU memory allocations and deallocations over time (e.g., `_record_memory_history`).

## 3. GPU Kernel Optimization
*   **FlashAttention-2**: An IO-aware attention algorithm that uses tiling, recomputation, and fusion to scale attention to long sequences with $O(N)$ memory.
*   **Triton**: A language and compiler for writing high-performance GPU kernels in Python, abstracting away CUDA C/C++ complexities.
*   **Block Pointer**: A Triton abstraction (`tl.make_block_ptr`) that simplifies managing multi-dimensional tiles and memory offsets in kernels.
*   **Occupancy**: The ratio of active warps on a SM (Streaming Multiprocessor) to the maximum possible warps, influenced by register and shared memory usage.
*   **Operator Fusion**: Combining multiple mathematical operations into a single GPU kernel to reduce expensive HBM (High Bandwidth Memory) reads and writes.
*   **Online Softmax**: A technique for computing softmax incrementally over tiles (using running max and sum) without materializing the full attention matrix.

## 4. Distributed Training
*   **Collective Communication**: Communication patterns involving multiple processes, such as `broadcast`, `all-reduce`, and `all-gather`.
*   **All-Reduce**: A collective operation that sums (or averages) tensors across all processes and distributes the final result back to all of them.
*   **Distributed Data Parallel (DDP)**: A parallelism strategy that shards data batches across GPUs, each maintaining a full model copy and synchronizing gradients.
*   **Gradient Bucketing**: Batching individual parameter gradients into larger "buckets" for `all-reduce` to minimize communication overhead.
*   **Asynchronous Communication (`async_op=True`)**: Executing communication calls in the background, allowing them to overlap with backward pass computation.
*   **Optimization State Sharding (ZeRO Stage 1)**: Partitioning optimizer states (like Adam moments) across GPUs to reduce per-rank memory consumption.

## 5. Advanced Parallelism (4D Mesh)
*   **Tensor Parallelism (TP)**: Sharding individual weight matrices and activations across multiple devices within a single layer.
*   **Pipeline Parallelism (PP)**: Partitioning model layers into stages and processing them in a pipeline fashion across devices.
*   **Fully-Sharded Data Parallel (FSDP)**: An evolution of ZeRO that shards parameters, gradients, and optimizer states across all participating GPUs.
*   **Expert Parallelism (EP)**: Sharding different experts in a Mixture-of-Experts (MoE) model across different devices.
*   **4D Mesh**: A conceptual grid representing the axes of parallelism (DP, FSDP/TP, PP, EP) used to coordinate large-scale training.
