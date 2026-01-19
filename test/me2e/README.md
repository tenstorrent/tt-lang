# ME2E Test Framework

Middle-end to end-to-end test framework for TTL MLIR generation and validation. Requires the tt-lang python
environment to be activated: `source build/env/activate`.

## Overview

This framework provides pytest infrastructure for testing TTL dialect operations starting with `ttl` dialect MLIR and
lowering all the way to C++, then executing with the TTNN JIT runtime.

**Important**: This framework does NOT test individual compiler passes. Pass transformations
are already comprehensively tested in `test/ttlang/` using lit/FileCheck.

## Testing Architecture and Workflow

The ME2E test framework follows a multi-stage pipeline architecture that mirrors the actual compilation and execution flow. Tests progress through ordered stages, with each stage producing artifacts consumed by subsequent stages.

### Test Architecture

The framework provides four types of tests:

1. Declarative parametrized tests (`test_compute_ops.py`):
   - Single parametrized test function covering all elementwise operations
   - Operations defined declaratively in `op_specs.py` as `ComputeOpSpec` entries
   - Tests 13 operations (add, sub, mul, max, exp, log, sqrt, rsqrt, tanh, abs, neg, relu, sigmoid)
   - Each operation tested with multiple configurations from `config_specs.py` (1x1, 2x2 grids)
   - All pipeline stages (build -> compile -> translate -> execute -> validate) executed in a single test function via `runner.py`
   - Uses temporary directories for kernel artifacts
   - This is the primary test approach for elementwise operations

2. Auto-generated class-based tests (`ops/test_binary.py`, `ops/test_unary.py`):
   - Test classes auto-generated from `TTLElementwiseOps.def` (tablegen)
   - Operations parsed from the `.def` file at module load time
   - Uses `OpTestBase` with ordered test stages (5 separate test methods with `@pytest.mark.order()` decorators)
   - Artifacts saved to `build/test/me2e/<TestClassName>/`
   - Provides class-based alternative to declarative tests

3. Custom MLIR tests (`ops/test_fused.py`):
   - Class-based tests using `FusedOpTestBase` (extends `E2ETestBase`)
   - Provides custom MLIR templates as strings for fused operations
   - Examples: exp(a+b), relu(a*b), sqrt(abs(a))
   - 5 separate test methods with `@pytest.mark.order()` decorators
   - Artifacts saved to `build/test/me2e/<TestClassName>/`

4. Builder validation tests (`ops/test_simple.py`):
   - Tests MLIR generation correctness without full pipeline
   - Verifies builder produces valid MLIR for different ops/shapes/dtypes
   - Does not execute on hardware

### Pipeline Stages

The declarative tests (`test_compute_ops.py`) execute all stages in a single function, while class-based tests (`ops/test_binary.py`, `ops/test_unary.py`, `ops/test_fused.py`) use separate ordered test methods. Both approaches follow the same 5-stage pipeline:

```
1. test_build_module (order=1)
   ├─ Input: Operation spec, configuration
   ├─ Output: module.mlir, inputs.pt, golden.pt
   └─ Validates: MLIR generation correctness

2. test_compile_to_ttkernel (order=2)
   ├─ Input: module.mlir
   ├─ Output: compiled_module.mlir
   └─ Validates: TTL -> TTKernel lowering

3. test_translate_to_cpp (order=3)
   ├─ Input: compiled_module.mlir
   ├─ Output: kernels/*.cpp (reader, compute, writer)
   └─ Validates: TTKernel -> C++ codegen

4. test_execute (order=4)
   ├─ Input: kernels/*.cpp, inputs.pt
   ├─ Output: result.pt
   └─ Validates: TTNN JIT execution

5. test_validate_golden (order=5)
   ├─ Input: result.pt, golden.pt
   ├─ Output: ULP comparison report
   └─ Validates: Numerical correctness
```

### Test Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Test Collection (pytest)                                    │
│ - Discovers test classes/functions                          │
│ - Applies pytest-order sorting (if available)               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Build Module                                       │
│ - Generate TTL MLIR via Python bindings                     │
│ - Create input tensors                                      │
│ - Compute golden reference (PyTorch)                        │
│ - Save: module.mlir, inputs.pt, golden.pt                   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Compile to TTKernel                                │
│ - Load module.mlir                                          │
│ - Run TTL pass pipeline (convert-ttl-to-compute, etc.)      │
│ - Lower to TTKernel dialect                                 │
│ - Save: compiled_module.mlir                                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Translate to C++                                   │
│ - Load compiled_module.mlir                                 │
│ - Extract reader/writer/compute kernels                     │
│ - Generate C++ source code                                  │
│ - Save: kernels/*.cpp                                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Execute on Device                                  │
│ - Load kernels/*.cpp and inputs.pt                          │
│ - Compile kernels via TTNN JIT                              │
│ - Execute using ttnn.generic_op                             │
│ - Save: result.pt                                           │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Validate Golden                                    │
│ - Load result.pt and golden.pt                              │
│ - Compare using ULP-based metrics                           │
│ - Assert max_ulp <= threshold                               │
└─────────────────────────────────────────────────────────────┘
```

## Running Tests

```bash
# Activate environment
source build/env/activate

# Install dev dependencies (including pytest-order for test ordering)
cmake --build build --target install-dev-deps

# All me2e tests
pytest test/me2e/

# Via CMake/Ninja
ninja -C build check-ttlang-me2e

# Just comparison utils
pytest test/me2e/test_utils.py

# Just MLIR builder validation
pytest test/me2e/ops/test_simple.py

# Demonstration/example tests
pytest test/me2e/examples/ -v -s
```

**Note**: Class-based tests (`ops/test_binary.py`, `ops/test_unary.py`, `ops/test_fused.py`) use `@pytest.mark.order()` to ensure proper execution order (e.g., `test_build_module` before `test_compile_to_ttkernel`). This requires `pytest-order` to be installed. The declarative tests (`test_compute_ops.py`) execute all stages in a single function and do not require test ordering. Install `pytest-order` via the `install-dev-deps` target or manually: `pip install -r dev-requirements.txt`.

## Adding New Operations

### Single-Op Tests

To add a new elementwise operation to `test_compute_ops.py`:

1. Add the op to `TTLElementwiseOps.def` (compiler side)
2. Add a `ComputeOpSpec` entry to `op_specs.py`:

```python
# In op_specs.py
COMPUTE_OPS = [
    # Add your new op
    ComputeOpSpec(
        name="my_op",                    # Operation name
        ttl_op="tile_my_op",            # TTL dialect op name
        arity=1,                         # 1 for unary, 2 for binary
        golden=torch.my_op,              # PyTorch reference function
        reader_type="unary",             # "unary" or "binary"
        input_range=(0.01, 10.0),       # Optional: constrain input domain
    ),
]
```

The operation will automatically be tested with all configurations in `config_specs.py`.

### Custom Fused Operations

For operations that combine multiple tile ops (not in the `.def` file), use
`FusedOpTestBase` in `test_fused.py`. These use MLIR string templates to build
`ttl.compute` regions with multiple fused tile operations:

```python
class TestExpAddFused(FusedOpTestBase):
    """Test exp(a + b) - fuses add and exp into single compute region."""

    OP_NAME = "exp_add"
    ARITY = 2
    INPUT_RANGE = (-2.0, 2.0)  # Limit to avoid exp overflow

    def torch_reference(self, a, b):
        return torch.exp(a + b)

    def get_mlir_template(self, config):
        # Returns MLIR with ttl.compute containing tile_add + tile_exp
        return f'''
        %result = ttl.compute ins(%a, %b) outs(%out) {{
        ^bb0(%a_tile, %b_tile, %out_tile):
          %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<...>
          %exp = ttl.tile_exp %sum : !ttcore.tile<...>
          ttl.yield %exp : !ttcore.tile<...>
        }}
        '''
```

The generated MLIR is compiled through passes, producing fused ttkernel ops:
```
ttkernel.add_binary_tile(%c0, %c1, %c0)
ttkernel.exp_tile(%c0)
```

See `test/me2e/ops/test_fused.py` for complete examples of:
- `TestExpAddFused`: exp(a + b)
- `TestReluMulFused`: relu(a * b)
- `TestSqrtAbsFused`: sqrt(abs(a))

## Test Artifacts

Test stages save intermediate artifacts to `build/test/me2e/<TestClassName>/`:

```
build/test/me2e/TestAdd/
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
test/me2e/
├── builder/                 # MLIR generation, compilation, and execution utilities
│   ├── ttl_builder.py       # Build TTL modules via Python bindings
│   ├── dm_threads.py        # Data movement thread templates
│   ├── pipeline.py          # Pass pipeline execution
│   ├── kernels.py           # Kernel translation utilities
│   ├── ttnn_runner.py       # TTNN device execution harness
│   ├── dtype_utils.py       # Dtype conversion utilities
│   └── device_arch.py       # Device architecture utilities
├── ops/                     # Operation tests
│   ├── __init__.py          # Auto-generation logic, base classes, op mappings
│   ├── test_simple.py       # MLIR builder validation tests
│   ├── test_binary.py       # Auto-generated binary op test classes
│   ├── test_unary.py        # Auto-generated unary op test classes
│   └── test_fused.py        # Custom fused operation tests
├── base.py                  # E2ETestBase defining pipeline stages
├── runner.py                # Declarative test runner (executes full pipeline)
├── test_compute_ops.py      # Declarative parametrized tests (main test suite)
├── op_specs.py              # Operation specifications (ComputeOpSpec entries)
├── config_specs.py          # Test configuration specifications (TestConfig entries)
├── config.py                # E2EConfig dataclass
├── utils.py                 # ULP comparison utilities
├── test_utils.py            # Comparison unit tests
├── conftest.py              # Pytest fixtures and configuration
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
| `test/ttlang/` | Compiler passes | lit/FileCheck | Pass transformations (MLIR -> MLIR) |
| `test/me2e/` | MLIR builder + execution | pytest | MLIR generation + hardware execution |
| `test/python/` | DSL front-end | lit/FileCheck | Python DSL -> MLIR |

Runtime dependencies are installed via `pyproject.toml` or `requirements.txt` when building tt-lang. Development dependencies (including `pytest-order`) are in `dev-requirements.txt` and can be installed via:

```bash
# Using CMake target (recommended)
cmake --build build --target install-dev-deps

# Or manually
pip install -r dev-requirements.txt
```

The `install-dev-deps` target installs dependencies into the virtual environment used by the project.
