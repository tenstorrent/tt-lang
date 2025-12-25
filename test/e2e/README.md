# E2E Test Framework

End-to-end test framework for TTL compilation and hardware execution. It requires the tt-lang python
environment to be activated before running the tests (e.g., `source build/env/activate`).

## Overview

This framework provides class-based pytest infrastructure for testing TTL dialect operations through the full pipeline: MLIR generation, compilation, kernel translation, hardware execution, and golden validation.

## Architecture

### Design Approach

The framework uses **class-based inheritance** to minimize boilerplate and maximize code reuse. Test classes inherit from specialized base classes that handle the full compilation and execution pipeline. This design enables:

- **Minimal test definitions** - Most tests are 2-3 lines of class attributes
- **Automatic pipeline execution** - Ordered stages (build → compile → execute → validate) run automatically
- **Graceful failure handling** - Downstream stages skip if prerequisites fail
- **Three input modalities** - Support for programmatic MLIR generation, manual MLIR files, and Python DSL
- **Flexible configuration** - Grid shapes, data types, and buffer settings via `TestConfig`
- **Auto-discovery** - Operations parsed from `TTLElementwiseOps.def` with torch reference functions

### Input Modalities

| Test Type | Input Method | Use Case | Location |
|-----------|--------------|----------|----------|
| **Ops** | Python bindings (programmatic) | Elementwise ops with minimal boilerplate | `test/e2e/ops/` |
| **MLIR** | Manual `.mlir` files | Custom kernels, complex data movement | `test/e2e/mlir/` |
| **DSL** | Python `@kernel` decorators | Python DSL validation | `test/e2e/dsl/` |

### Directory Structure

```
test/e2e/
├── base.py              # E2ETestBase with ordered pipeline stages
├── conftest.py          # Pytest fixtures and markers
├── config.py            # TestConfig and grid shapes
├── utils.py             # Comparison functions and exceptions
├── pipeline.py          # MLIR pass pipeline execution
├── kernels.py           # KernelSpec with N data movement threads
├── ttl_builder.py       # TTL MLIR module builder
├── ops_gen.py           # Auto-parse TTLElementwiseOps.def
├── ops/                 # Elementwise operation tests
│   ├── __init__.py      # OpTestBase, UnaryOpTestBase, BinaryOpTestBase
│   ├── test_binary.py   # Binary op tests (add, sub, mul, max)
│   └── test_unary.py    # Unary op tests (exp, log, sqrt, etc.)
├── mlir/                # Manual MLIR file tests
│   ├── __init__.py      # MLIRFileTestBase
│   └── inputs/          # .mlir test files
└── dsl/                 # Python DSL tests
    └── __init__.py      # DSLTestBase
```

## Test Types

### 1. Operation Tests (Minimal Class Definitions)

Test elementwise operations with minimal boilerplate:

```python
# test/e2e/ops/test_binary.py
from . import BinaryOpTestBase

class TestAdd(BinaryOpTestBase):
    OP_STR = "add"

class TestSub(BinaryOpTestBase):
    OP_STR = "sub"
```

Override class attributes to customize:

```python
class TestSqrt(UnaryOpTestBase):
    OP_STR = "sqrt"
    INPUT_RANGE = (0.01, 10.0)  # Domain constraint
    ULP_THRESHOLD = 1.0         # Stricter ULP threshold
```

**Note**: ULP thresholds are auto-computed from dtype:
- `torch.bfloat16` / `torch.float16`: 2.0 ULP
- `torch.float32` / `torch.float64`: 1.0 ULP
- Integer types: 0.0 ULP (exact)

### 2. MLIR File Tests

Test manually written MLIR files:

```python
from test.e2e.mlir import MLIRFileTestBase
from pathlib import Path

class TestCustomKernel(MLIRFileTestBase):
    MLIR_PATH = Path("test/e2e/mlir/inputs/custom_kernel.mlir")

    # Override for custom validation
    def test_validate_golden(self):
        # Custom golden comparison logic
        ...
```

### 3. DSL Tests

Test Python DSL kernels:

```python
from test.e2e.dsl import DSLTestBase
from ttlang.d2m_api import kernel

class TestRuntimeAdd(DSLTestBase):
    @staticmethod
    @kernel(grid=(1, 1))
    def add_kernel(lhs, rhs, out):
        # DSL code
        ...

    DSL_FUNC = add_kernel
```

## Pipeline Stages

Tests execute in ordered stages with dependency checking:

1. **test_build_module()** - Build or load TTL MLIR module
2. **test_compile_to_ttkernel()** - Run TTL-to-TTKernel passes
3. **test_translate_to_cpp()** - Translate to C++ kernel sources
4. **test_execute()** - Execute kernels on device
5. **test_validate_golden()** - Compare result with golden

If a stage fails, downstream stages skip automatically via `_check_cache_dependencies()`.

## Running Tests

```bash
# All E2E tests
pytest test/e2e/

# Specific test class
pytest test/e2e/ops/test_binary.py::TestAdd

# With MLIR dump
pytest test/e2e/ --dump-mlir

# Specific hardware target
pytest test/e2e/ --sys-desc=/path/to/system.ttsys

# Generate JUnit XML report
pytest test/e2e/ --junit-xml=build/test-results/e2e.xml
```

## Markers

### Target Exclusions

Skip tests on specific hardware:

```python
@pytest.mark.skip_target("n150", reason="Not supported on N150")
def test_something(self):
    ...

@pytest.mark.only_target("n300", "p150")
def test_something_else(self):
    ...
```

### Expected Failures

Mark tests expected to fail:

```python
@pytest.mark.xfail(reason="sqrt lowering not yet implemented", strict=True)
def test_compile_to_ttkernel(self):
    super().test_compile_to_ttkernel()
```

Options:
- `reason="..."` - Explanation for xfail
- `strict=True` - Fail if test unexpectedly passes
- `raises=ExceptionType` - Only xfail for specific exception

## Configuration

### TestConfig

Configure grid shape, dtype, and buffer settings:

```python
from test.e2e.config import TestConfig, BufferType

config = TestConfig(
    grid_shape=(4, 4),           # 4x4 tiles
    dtype=torch.bfloat16,        # Data type
    buffer_factor=2,             # Double buffering
    buffer_type=BufferType.DRAM, # DRAM vs L1
)
```

Predefined configs:
- `SMOKE_CONFIGS` - Minimal (1x1 tile)
- `CONFIGS` - Standard test suite
- `SMALL_GRIDS`, `MEDIUM_GRIDS`, `LARGE_GRIDS` - Grid shape sets

## Cache Management

The `CACHE` dictionary passes data between pipeline stages:

```python
class E2ETestCache(TypedDict):
    module: Module                    # TTL MLIR module
    compiled_module: Module           # After pass pipeline
    noc_kernels: List[KernelSpec]     # Data movement kernels
    compute_kernel: KernelSpec        # Compute kernel
    torch_inputs: List[torch.Tensor]  # Input tensors
    device_inputs: List[Any]          # On-device tensors
    output_tensor: Any                # Device output
    golden: torch.Tensor              # Expected result
    result: torch.Tensor              # Actual result
```

Access in tests:

```python
def test_something(self):
    self._check_cache_dependencies(["module", "torch_inputs"])
    module = self.CACHE["module"]
    inputs = self.CACHE["torch_inputs"]
    # ... use cached data
```

## Auto-Generated Operations

Operations are auto-discovered from `include/ttlang/Dialect/TTL/TTLElementwiseOps.def`:

```cpp
// Binary operations
TTL_BINARY_TILE_OP(Add, AddTileOp)
TTL_BINARY_TILE_OP(Sub, SubTileOp)

// Unary operations
TTL_UNARY_TILE_OP(Exp, ExpTileOp)
TTL_UNARY_TILE_OP(Sqrt, SqrtTileOp)
```

Torch reference functions are mapped in `ops_gen.py`:

```python
OP_TORCH_MAP = {
    "add": torch.add,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    # ...
}
```

## Extending the Framework

### Add New Operation Test

1. Add to `TTLElementwiseOps.def` (if not present)
2. Add torch reference to `OP_TORCH_MAP` in `ops_gen.py`
3. Create test class:

```python
class TestNewOp(UnaryOpTestBase):
    OP_STR = "newop"
```

### Add Custom Pipeline Stage

Override or add stages in your test class:

```python
class TestCustomPipeline(OpTestBase):
    @pytest.mark.order(after="test_compile_to_ttkernel")
    def test_custom_optimization(self):
        self._check_cache_dependencies(["compiled_module"])
        # Custom pass pipeline
        self._test_pipeline_step(
            "compiled_module",
            "optimized_module",
            ["custom-pass-1", "custom-pass-2"]
        )
```

### Override Golden Validation

Provide custom comparison logic:

```python
class TestCustomValidation(OpTestBase):
    def test_validate_golden(self):
        self._check_cache_dependencies(["result"])
        result = self.CACHE["result"]
        # Custom validation
        assert custom_check(result), "Custom validation failed"
```

## Comparison Utilities

### compare_tensors()

```python
from test.e2e.utils import compare_tensors

# Auto-computed ULP threshold
comparison = compare_tensors(golden, calculated)

# Override threshold
comparison = compare_tensors(golden, calculated, ulp_threshold=1.0)

assert comparison.passed, comparison.message
print(f"Max ULP: {comparison.max_ulp:.2f}")
print(f"Mean ULP: {comparison.mean_ulp:.2f}")
```

**ULP (Units of Least Precision)**: Measures error in representable floating-point values. Hardware-accurate, scale-independent, and dtype-aware. Based on Goldberg's "What Every Computer Scientist Should Know About Floating-Point Arithmetic".

## Kernel Model

Supports N data movement (NOC) threads + 1 compute thread:

```python
from test.e2e.kernels import KernelSpec, ThreadType

# NOC kernels (data movement)
noc_kernels = [
    KernelSpec(name="reader_0", thread_type=ThreadType.NOC, source=cpp_code),
    KernelSpec(name="reader_1", thread_type=ThreadType.NOC, source=cpp_code),
    KernelSpec(name="writer", thread_type=ThreadType.NOC, source=cpp_code),
]

# Compute kernel
compute_kernel = KernelSpec(
    name="compute_add",
    thread_type=ThreadType.COMPUTE,
    source=cpp_code
)
```

## Design Principles

1. **Minimal test definitions** - One line per operation test
2. **Class-based inheritance** - Share common behavior via base classes
3. **Ordered execution** - Pipeline stages run in dependency order
4. **Graceful failure** - Downstream tests skip if prerequisites fail
5. **Declarative markers** - Use pytest markers for xfail/skip
6. **Auto-generation** - Parse operation definitions from source
7. **Extensibility** - Easy to add custom stages and validation

## Line Count

Total: **1,553 lines** (32% reduction from original 2,274 lines)

## Requirements

- pytest >= 7.0
- pytest-order >= 1.0.0
- ttnn (for hardware execution)
- torch (for golden reference)
