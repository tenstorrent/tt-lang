---
description: Import and translate a CUDA, Triton, or PyTorch kernel to TT-Lang DSL
argument-hint: <kernel-file-or-code>
---

## Task

Translate the provided kernel to a TT-Lang DSL program with equivalent behavior.

## Input

$ARGUMENTS

## Process

1. **Analyze the source kernel**
   - Identify the kernel type (CUDA, Triton, or PyTorch)
   - Extract the computation pattern and data flow
   - Note memory access patterns and synchronization points

2. **Create semantic translation**
   - Map GPU concepts to TT-Lang equivalents:
     - Thread blocks -> Tensix cores
     - Shared memory -> L1 circular buffers
     - Global memory -> DRAM with DMA transfers
     - Warps/waves -> Tile-level operations
   - Preserve the computational semantics, not the implementation details

3. **Generate TT-Lang kernel**
   - Write the `@pykernel_gen` decorated function
   - Use appropriate TT-Lang primitives:
     - `TensorBlock` for tile-level data
     - `MemTx` for memory transfers
     - `CircularBuffer` for inter-thread communication
     - `Semaphore` for multi-core synchronization

4. **Validate and iterate**
   - Run the simulator to verify correctness
   - Compare outputs against the original kernel
   - Fix any semantic differences until behavior matches

## Output

Provide the translated TT-Lang kernel with:
- Clear comments explaining the mapping from source concepts
- A test harness to verify equivalence
- Notes on any limitations or differences in behavior
