# tt-lang Examples

This directory contains example kernels demonstrating the tt-lang DSL for writing custom data movement and compute kernels.

## Running Examples

### With the TT-Lang Simulator (Recommended for Development)

Most examples can be run with the tt-lang simulator, which doesn't require hardware:

```bash
# Activate the environment
source build/env/activate

# Run any example with the simulator
python -m sim.ttlang_sim examples/<example_name>.py
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
| `eltwise_pipe.py` | Fused element-wise ops using pipe multicasting | ✓ | ✓ |
| `eltwise_pipe_core3.py` | Variant of eltwise_pipe with different core config | ✓ | ✓ |

### Broadcasting

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `broadcast.py` | Column vector broadcast during element-wise op | ✓ | ✗ |
| `broadcast_demo.py` | Well-documented scalar broadcast example | ✓ | ✗ |
| `general_broadcast.py` | Adaptive broadcasting based on tensor shapes | ✓ | ✗ |

### Matrix Multiplication

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `singlecore_matmul.py` | Single-core matrix multiplication | ✓ | ✗ |
| `multicore_matmul.py` | Multi-core matmul with work distribution | ✓ | ✗ |

### Demo/Tutorial

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `demo_one.py` | Comprehensive demo with auto grid and bounds checking | ✓ | ✓ |

### Error Examples (Negative Tests)

These examples intentionally fail to demonstrate error handling:

| Example | Description | Expected Error |
|---------|-------------|----------------|
| `eltwise_add_error.py` | Shape mismatch in copy operation | "Tensor shape does not match Block shape" |
| `copy_lock_error.py` | Block access before wait() completes | "Cannot write to Block: Block has no access" |

## Metal Examples

The `metal_examples/` directory contains paired implementations comparing tt-lang with raw Metal reference implementations.

| Example | Description | Sim | HW |
|---------|-------------|:---:|:--:|
| `singlecore_matmul/ttlang/` | Single-core matmul in tt-lang | ✓ | ✗ |
| `multicore_matmul/ttlang/` | Multi-core matmul in tt-lang | ✓ | ✗ |
| `multicore_reuse_matmul/ttlang/` | Reuse-optimized matmul in tt-lang | ✗ | ✗ |

The `metal/` subdirectories contain reference Metal implementations for comparison.

## Testing

All examples (except error examples) are tested in CI via the tt-lang simulator:

```bash
# Run all example tests
pytest test/sim/test_examples.py -v

# Run a specific example test
pytest test/sim/test_examples.py::test_example_cli[eltwise_add.py] -v
```
