# tt-lang Examples

This directory contains example kernels demonstrating the tt-lang DSL for writing custom data movement and compute kernels.

## Running Examples

### With the TT-Lang Simulator (Recommended for Development)

Most examples can be run with the tt-lang simulator, which doesn't require hardware:

```bash
# Activate the environment
source build/env/activate

# Run any example with the simulator
tt-lang-sim examples/<example_name>.py
```

### On Hardware

Certain examples can also be run directly on Tenstorrent hardware (requires device access):

```bash
source build/env/activate
python examples/<example_name>.py
```

## Example Categories

### Element-wise Operations

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `eltwise_add.py` | Element-wise addition of two tensors | ✓ | ✓ |
| `eltwise_pipe.py` | Fused element-wise ops using pipe multicasting | ✓ | ✗ |
| `eltwise_pipe_node3.py` | Variant of eltwise_pipe with different node config | ✓ | ✗ |

### Broadcasting

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `broadcast.py` | Column vector broadcast during element-wise op | ✓ | ✗ |
| `broadcast_demo.py` | Well-documented scalar broadcast example | ✓ | ✓ |
| `general_broadcast.py` | Adaptive broadcasting based on tensor shapes | ✓ | ✗ |

### Matrix Multiplication

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `singlenode_matmul.py` | Single-node matrix multiplication | ✓ | ✗ |
| `multinode_matmul.py` | Multi-node matmul with work distribution | ✓ | ✗ |

### Demo/Tutorial

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `demo_one.py` | Comprehensive demo with auto grid and bounds checking | ✓ | ✓ |

### Error Examples (Negative Tests)

The `errors/` subdirectory contains examples with intentionally incorrect or risky code. They demonstrate how the simulator reports mistakes (shape checks, dataflow locks, deadlocks). Exact wording changes over time; tests in `test/sim/test_examples.py` pin the important substrings.

| Example | Description | Expected outcome |
|---------|-------------|------------------|
| `errors/eltwise_add_error.py` | Copy tile count mismatch (single tile into a multi-tile block) | Failure with a shape mismatch message (tensor vs block tile counts) and a source location on the bad `copy` call |
| `errors/copy_lock_error.py` | Store into a block while it is still a copy destination (before waiting on that copy) | Failure with NAW / copy-destination lock wording on `this buffer block`; diagnostics include the failing line and a **Where:** line pointing at the `copy(..., block)` callsite |
| `errors/copy_source_lock_error.py` | Store into a block while it is still a live copy *source* (ROR, before waiting on `copy(block, ...)`) | Failure with ROR / copy-source wording; **Where:** points at the `copy(block, tensor)` callsite |
| `errors/eltwise_add_deadlock.py` | Same layout as `eltwise_add.py` but read path uses `wait()` on producer buffers instead of `reserve()`, so nothing fills them | Failure with deadlock detection (`Deadlock detected: all generators blocked`) and blocked-kernel diagnostics |
| `errors/max_dfbs_warning.py` | Allocates more DataflowBuffers than the default hardware limit | **Warning** (not fatal): `UserWarning` about the DFB limit; script still exits successfully |

## Metal Examples

The `metal_examples/` directory contains paired implementations comparing tt-lang with raw Metal reference implementations.

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `singlenode_matmul/ttlang/` | Single-node matmul in tt-lang | ✓ | ✗ |
| `multinode_matmul/ttlang/` | Multi-node matmul in tt-lang | ✓ | ✗ |
| `multinode_reuse_matmul/ttlang/` | Reuse-optimized matmul in tt-lang | ✓ | ✗ |

The `metal/` subdirectories contain reference Metal implementations for comparison.

## Testing

Examples under `examples/` (including those under `examples/errors/`) are exercised by the simulator test suite:

```bash
# Run all example tests
pytest test/sim/test_examples.py -v

# Run a specific example test
pytest test/sim/test_examples.py::test_example_cli[eltwise_add.py] -v
```

Note: `check-ttlang-all` does not include `pytest test/sim`; see `test/TESTING.md` for simulator test scope.
