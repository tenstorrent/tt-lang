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

### 3. Auto-Generated Op Tests (`ops/test_binary.py`, `ops/test_unary.py`)

Test classes are **auto-generated** from `TTLElementwiseOps.def`:
- **4 binary ops**: add, sub, mul, max
- **9 unary ops**: exp, log, sqrt, rsqrt, tanh, abs, neg, relu, sigmoid

Adding a new op to the `.def` file automatically creates its test class.

### 4. Builder Validation Tests (`ops/test_simple.py`)

8 tests verifying MLIR generation edge cases:
- Binary ops (add, sub, mul)
- Unary ops (exp, sqrt, neg)
- Different grid shapes (1x1, 2x2, 4x4)
- Different dtypes (bf16, f32)

### 5. Example/Demonstration Tests (`examples/test_add_demo.py`)

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

## Auto-Generated Op Tests

Test classes for all elementwise operations are **automatically generated** from
`include/ttlang/Dialect/TTL/TTLElementwiseOps.def`. This ensures tests stay in
sync with the compiler dialect definitions.

When the `.def` file contains:
```cpp
TTL_BINARY_TILE_OP(Add, AddTileOp)
TTL_UNARY_TILE_OP(Exp, ExpTileOp)
```

The framework automatically creates `TestAdd` and `TestExp` classes.

### Adding a New Op

To add tests for a new op:

1. Add the op to `TTLElementwiseOps.def` (compiler side)
2. Add the torch reference to `OP_TORCH_MAP` in `ops/__init__.py`
3. Done! Test classes are generated automatically.

### Custom Torch Reference Functions

For ops without a direct torch equivalent, use a lambda or custom function:

```python
# In ops/__init__.py
OP_TORCH_MAP: Dict[str, Callable[..., Tensor]] = {
    # Standard mappings
    "add": torch.add,
    "exp": torch.exp,

    # Custom mapping for op with different semantics
    "custom_add_one": lambda x: x + 1.0,

    # Op that requires multiple torch ops
    "fused_mul_add": lambda a, b, c: a * b + c,

    # Op with special handling
    "safe_div": lambda a, b: torch.where(b != 0, a / b, torch.zeros_like(a)),
}
```

### Domain Constraints

Some ops require specific input ranges (e.g., log requires positive inputs).
Specify these in `OP_INPUT_RANGES`:

```python
# In ops/__init__.py
OP_INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "log": (0.01, 10.0),    # log requires positive inputs
    "sqrt": (0.01, 10.0),   # sqrt requires positive inputs
    "rsqrt": (0.01, 10.0),  # rsqrt requires positive inputs
    "acos": (-1.0, 1.0),    # acos domain is [-1, 1]
}
```

### Overriding Generated Test Classes

For special cases, override the generated class:

```python
# In test_unary.py or test_binary.py
from . import UnaryOpTestBase, GENERATED_OP_TESTS

# Import auto-generated tests
for name, cls in GENERATED_OP_TESTS.items():
    globals()[name] = cls

# Override specific op with custom behavior
class TestSpecialOp(UnaryOpTestBase):
    OP_STR = "special_op"
    INPUT_RANGE = (0.0, 100.0)
    ULP_THRESHOLD = 10.0  # Looser tolerance for this op

    # Custom golden computation
    @pytest.fixture(scope="class")
    def torch_op(self):
        def custom_golden(x):
            return torch.special.some_function(x)
        return custom_golden
```

## Test Artifacts

Test stages save intermediate artifacts to `build/test/e2e/<TestClassName>/`:

```
build/test/e2e/TestAdd/
├── module.mlir           # High-level TTL ops (from test_build_module)
├── compiled_module.mlir  # Lowered to TTKernel ops (from test_compile_to_ttkernel)
└── inputs.pt             # PyTorch input tensors for golden comparison
```

**Example `module.mlir`** (before passes):
```mlir
func.func @compute_add(...) {
  %0 = ttl.bind_cb {cb_index = 0, ...}
  %1 = ttl.attach_cb %arg0, %0 : ...
  %2 = ttl.add %1, %3 : ...
  return %2 : ...
}
```

**Example `compiled_module.mlir`** (after passes):
```mlir
func.func @compute_add(...) {
  ttkernel.copy_tile(%cb0, %idx, %c0)
  ttkernel.add_binary_tile(%c0, %c1, %c0)
  ttkernel.pack_tile(%c0, %cb_out, %c0, false)
  ...
}
```

These artifacts are useful for debugging and can be processed manually with
`ttlang-opt` or other tools.

## Directory Structure

```
test/e2e/
├── ops/
│   ├── __init__.py          # Auto-generation logic, base classes
│   ├── test_simple.py       # MLIR builder validation (8 tests)
│   ├── test_binary.py       # Auto-generated binary op tests
│   └── test_unary.py        # Auto-generated unary op tests
├── examples/
│   └── test_add_demo.py     # Full workflow demonstration (5 tests)
├── utils.py                 # ULP comparison utilities
├── test_utils.py            # Comparison unit tests (11 tests)
├── ttl_builder.py           # MLIR module builder
├── config.py                # E2EConfig dataclass
├── base.py                  # E2ETestBase with pipeline stages
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

### End-to-End Execution Tests

The framework is structured to support full execution tests. Test classes are
auto-generated from `TTLElementwiseOps.def`:

```python
# In TTLElementwiseOps.def:
TTL_BINARY_TILE_OP(Add, AddTileOp)

# Automatically generates TestAdd with ordered test stages:
# 1. test_build_module     - Generate MLIR
# 2. test_compile_to_ttkernel - Compile through passes
# 3. test_translate_to_cpp - Generate kernel sources
# 4. test_execute          - Run on hardware
# 5. test_validate_golden  - Compare with torch golden
```

But execution requires:
- Kernel compilation infrastructure
- Hardware execution via ttnn
- Result extraction and comparison

### Test Result Caching (Optional Optimization)

A caching mechanism can be added to avoid recomputing intermediate results:
- Cache MLIR modules, compiled modules, execution results between test stages
- Allow running later stages (e.g., `test_compile`) without re-running earlier stages
- Class-scoped cache shared across test methods within each test class
- Improves iteration speed during development

**Design Note**: Caching is optional. Tests should work independently first,
then caching can be added as a performance optimization.

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
