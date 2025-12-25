# E2E Test Framework

End-to-end test framework for TTL MLIR generation and validation. Requires the tt-lang python
environment to be activated: `source build/env/activate`.

## Overview

This framework provides pytest infrastructure for testing TTL dialect operations. The focus is on:

1. **MLIR Builder Validation**: Verify that programmatic MLIR generation produces correct TTL dialect code
2. **Comparison Utilities**: ULP-based tensor comparison for golden validation
3. **Future: End-to-End Execution**: Full pipeline from MLIR → compile → execute → validate

**Important**: This framework does NOT test compiler passes. Pass transformations (e.g., `convert-ttl-to-compute`) are already comprehensively tested in `test/ttlang/` using lit/FileCheck.

## What's Working

### 1. MLIR Builder (`ttl_builder.py`)

Programmatically generates TTL dialect MLIR:

```python
from test.e2e.ttl_builder import build_ttl_module
from test.e2e.config import E2EConfig

config = E2EConfig(grid_shape=(2, 2), dtype=torch.bfloat16)
inputs = [torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)]
module = build_ttl_module("add", 2, config, inputs)
```

Generates:
```mlir
func.func @compute_add(%arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>,
                       %arg1: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<...>
  %1 = ttl.attach_cb %arg0, %0 : ...
  %2 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<...>
  %3 = ttl.attach_cb %arg1, %2 : ...
  %4 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<...>
  %5 = ttl.add %1, %3 : ...
  return %5 : ...
}
```

### 2. Comparison Utilities (`utils.py`, `test_utils.py`)

ULP-based tensor comparison with automatic dtype-specific thresholds:

```python
from test.e2e.utils import compare_tensors

result = compare_tensors(golden, calculated)
assert result.passed, result.message
# Reports max_ulp, mean_ulp in error messages
```

- **11/11 unit tests passing** for comparison logic
- Hardware-accurate (ULP-based, not epsilon-based)
- Scale-independent and dtype-aware

### 3. Builder Validation Tests (`ops/test_simple.py`)

8 tests verifying MLIR generation:
- Binary ops (add, sub, mul)
- Unary ops (exp, sqrt, neg)
- Different grid shapes (1x1, 2x2, 4x4)
- Different dtypes (bf16, f32)

### 4. Example/Demonstration Tests (`examples/test_add_demo.py`)

Complete workflow demonstration for the `add` operation:
- MLIR generation
- Torch reference validation
- Configuration variations

These verbose tests serve as templates and documentation for writing new tests.

## Running Tests

```bash
# Activate environment
source build/env/activate

# All e2e tests
pytest test/e2e/

# Via CMake/Ninja
ninja -C build check-ttlang-e2e

# Just comparison utils
pytest test/e2e/test_utils.py

# Just MLIR builder validation
pytest test/e2e/ops/test_simple.py

# Demonstration/example tests
pytest test/e2e/examples/ -v -s
```

## Directory Structure

```
test/e2e/
├── ops/
│   ├── __init__.py          # Base classes (for future execution tests)
│   ├── test_simple.py       # MLIR builder validation (8 tests)
│   ├── test_binary.py       # Binary ops (add, sub, mul, max) - minimal syntax
│   └── test_unary.py        # Unary ops (exp, log, sqrt, etc.) - minimal syntax
├── examples/
│   └── test_add_demo.py     # Full workflow demonstration (5 tests)
├── utils.py                 # ULP comparison utilities
├── test_utils.py            # Comparison unit tests (11 tests)
├── ttl_builder.py           # MLIR module builder
├── config.py                # E2EConfig dataclass
├── ops_gen.py               # Parse TTLElementwiseOps.def
├── base.py                  # Base classes for future execution tests
├── conftest.py              # Pytest fixtures
└── README.md                # This file
```

## Configuration

The `E2EConfig` dataclass defines test parameters:

```python
@dataclass(frozen=True)
class E2EConfig:
    grid_shape: Tuple[int, int] = (2, 2)  # Tiles (rows, cols)
    dtype: torch.dtype = torch.bfloat16
    buffer_factor: int = 1                 # 1=single, 2=double buffer
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED
    buffer_type: BufferType = BufferType.DRAM
```

## Design Principles

1. **Isolate Middle-End**: Test MLIR generation and transformations without DSL front-end
2. **Avoid Redundancy**: Don't retest what lit tests already cover comprehensively
3. **Focus on Value**: Test MLIR builder correctness and (future) end-to-end execution
4. **Minimal Code**: ~500 LOC for builder + comparison utilities + tests

## Future Work

The framework is structured to support end-to-end execution tests:

```python
class TestAdd(BinaryOpTestBase):
    OP_STR = "add"  # 2 lines = full test

# Base class will handle:
# 1. Generate inputs
# 2. Build MLIR module
# 3. Compile through passes
# 4. Execute on hardware
# 5. Compare with torch golden
```

But execution requires:
- Kernel compilation infrastructure
- Hardware execution via ttnn
- Result extraction and comparison

## Relationship to Other Tests

| Test Suite | Purpose | Tool | What It Tests |
|------------|---------|------|---------------|
| `test/ttlang/` | Compiler passes | lit/FileCheck | Pass transformations (MLIR → MLIR) |
| `test/e2e/` | MLIR builder + execution | pytest | MLIR generation + (future) hardware execution |
| `test/python/` | DSL front-end | lit/FileCheck | Python DSL → MLIR |

**Key Insight**: Pass testing belongs in lit tests. E2E tests should focus on MLIR builder validation and end-to-end execution, not redundantly testing passes.

## Requirements

- Python 3.11+
- pytest>=7.0
- torch
- ttmlir Python bindings
- ttlang Python bindings

Installed via `pyproject.toml` or `dev-requirements.txt` when building tt-lang.
