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
   - Operations auto-generated from `TTLElementwiseOps.def` in `op_specs.py`
   - Tests all operations defined in the `.def` file (currently 13: add, sub, mul, max, exp, log, sqrt, rsqrt, tanh, abs, neg, relu, sigmoid)
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
   - Class-based tests using `FusedOpTestBase` (extends `ME2ETestBase`)
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
   │          kernels/kernel_metadata.json (tensor indices)
   └─ Validates: TTKernel -> C++ codegen

4. test_execute (order=4)
   ├─ Input: kernels/*.cpp, kernels/kernel_metadata.json, inputs.pt
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
│ - Extract tensor indices from ttl.crta_indices attributes   │
│ - Generate C++ source code (unmodified compiler output)     │
│ - Save: kernels/*.cpp, kernels/kernel_metadata.json         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Execute on Device                                  │
│ - Load kernels/*.cpp, kernel_metadata.json, inputs.pt       │
│ - Build kernel descriptors using shared kernel_runner       │
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
2. Add the torch reference to `OP_TORCH_MAP` in `ops/__init__.py` (if not already present)
3. Optionally add input range constraints to `OP_INPUT_RANGES` in `ops/__init__.py` (for ops with domain constraints)

The operation will automatically be tested with all configurations in `config_specs.py`. The `COMPUTE_OPS` list in `op_specs.py` is auto-generated from the `.def` file, so no manual `ComputeOpSpec` entry is needed.

**Example**:
```python
# In ops/__init__.py
OP_TORCH_MAP: Dict[str, Callable[..., Tensor]] = {
    # ... existing ops ...
    "my_op": torch.my_op,  # Add torch reference
}

OP_INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    # ... existing constraints ...
    "my_op": (0.01, 10.0),  # Optional: if domain constraints are needed
}
```

The `ttl_op` name (e.g., `tile_my_op`) and `reader_type` (unary/binary) are automatically derived from the `.def` file.

### Custom Fused Operations

For operations that combine multiple tile ops (not in the `.def` file), there
are two approaches:

**1. Programmatic approach using `build_e2e_module_mlir_custom`:**

```python
from test.me2e.builder import build_e2e_module_mlir_custom
import ttl.dialects.ttl as ttl

# exp(a + b) - fused add and exp
mlir = build_e2e_module_mlir_custom(
    name="compute_exp_add",
    arity=2,
    num_outputs=1,
    config=config,
    compute_fn=lambda inputs, builder: [
        ttl.exp(
            builder.tile_tensor_type,
            ttl.add(builder.tile_tensor_type, inputs[0], inputs[1], loc=builder.loc),
            loc=builder.loc,
        )
    ],
)
```

**2. Class-based approach using `FusedOpTestBase`:**

For test classes, use `FusedOpTestBase` in `test_fused.py` with MLIR string
templates to build `ttl.compute` regions:

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
├── module.mlir                      # High-level TTL ops (from test_build_module)
├── compiled_module.mlir             # Lowered to TTKernel ops (from test_compile_to_ttkernel)
├── inputs.pt                        # PyTorch input tensors for golden comparison
├── golden.pt                        # Expected output from PyTorch reference
├── result.pt                        # Actual output from device execution
└── kernels/                         # Generated C++ kernels (from test_translate_to_cpp)
    ├── reader_binary.cpp            # Data movement reader kernel
    ├── writer.cpp                   # Data movement writer kernel
    ├── compute_add.cpp              # Compute kernel
    └── kernel_metadata.json         # Kernel metadata (tensor indices)
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

## Kernel Execution Architecture

### Shared Kernel Runner

The ME2E tests use the same kernel execution infrastructure as the Python DSL, ensuring that compiler-generated C++ runs unmodified on device. The shared `kernel_runner` module (`python/ttl/kernel_runner.py`) provides:

- **Single source of truth**: Both Python DSL (`CompiledTTNNKernel`) and ME2E tests use the same argument-building logic
- **No shimming required**: Generated C++ is written unmodified - no regex transformations or post-processing
- **Metadata-driven execution**: Tensor indices extracted from MLIR attributes (`ttl.crta_indices`) drive argument construction

**Key components**:
- `KernelSpec`: Dataclass specifying kernel path, thread type, and tensor indices
- `build_tensor_accessor_args()`: Builds compile-time args for tensor accessors
- `build_kernel_descriptors()`: Builds kernel descriptors with proper arguments
- `build_cb_descriptors()`: Builds circular buffer descriptors
- `run_kernel_on_device()`: Main entry point for kernel execution

**Execution flow**:
1. Extract `ttl.crta_indices` from compiled MLIR (which tensors each kernel accesses)
2. Save metadata to `kernel_metadata.json` alongside generated C++
3. Load metadata during execution to build proper `common_runtime_args`
4. Use shared `kernel_runner` to execute via `ttnn.generic_op`

This architecture eliminates the previous regex-based shimming approach (73 lines removed) and ensures ME2E tests validate the same code path as production.

## Builder Architecture

The MLIR builder uses a layered class hierarchy to eliminate boilerplate and
provide reusable building blocks for thread construction.

```
ThreadBuilder (base)              StringBasedThreadBuilder (base)
    │                                     │
    └── ComputeThreadBuilder              └── DMThreadBuilder
```

### Base Classes

**`ThreadBuilder`** - Python MLIR bindings based builder for compute threads:
- Type factories: `tile_type`, `cb_type`, `tile_tensor_type`
- CB operations: `_bind_cb`, `_cb_reserve`, `_cb_wait`, `_cb_push`, `_cb_pop`, `_attach_cb`
- Loop construct: `_with_tile_loop(body_fn)` handles single/multi-tile iteration
- Thread builder: `_build_compute_thread(name, input_cbs, output_cbs, compute_fn)`

**`StringBasedThreadBuilder`** - String-based builder for DM threads (required for layout types):
- Type strings: `dram_tensor_type_str`, `cb_type_str`, `slice_type_str` (parameterized by config)
- Loop generation: `_generate_loop_start()` returns loop code and index variables
- Transfer helpers: `_read_to_cb_str()`, `_write_from_cb_str()`
- Layout generation: Uses `config.buffer_type` (DRAM/L1) and `config.memory_layout` (interleaved/sharded)

### Thread Builders

**`DMThreadBuilder`** - Minimal data movement thread builder:
```python
dm_builder = DMThreadBuilder(config)
reader_mlir = dm_builder.build_reader(num_inputs=2)  # Binary reader
writer_mlir = dm_builder.build_writer(output_cbs=[2])  # Single output
```

**`ComputeThreadBuilder`** - Minimal compute thread builder:
```python
# Single elementwise op
builder.build_compute("add", arity=2)

# Custom/fused ops via callback
builder.build_compute_custom(
    name="compute_exp_add",
    input_cbs=[0, 1],
    output_cbs=[2],
    compute_fn=lambda inputs: [ttl.exp(tt, ttl.add(tt, inputs[0], inputs[1]))],
)
```

### Building ME2E Modules

Use the high-level `build_e2e_module_mlir()` function:
```python
from test.me2e.builder import build_e2e_module_mlir
from test.me2e.config import E2EConfig

config = E2EConfig(grid_shape=(2, 2))
mlir = build_e2e_module_mlir("add", arity=2, config=config)
```

For custom/fused operations:
```python
from test.me2e.builder import build_e2e_module_mlir_custom

mlir = build_e2e_module_mlir_custom(
    name="compute_exp_add",
    arity=2,
    num_outputs=1,
    config=config,
    compute_fn=lambda inputs, builder: [
        ttl.exp(builder.tile_tensor_type,
                ttl.add(builder.tile_tensor_type, inputs[0], inputs[1]))
    ],
)
```

## Directory Structure

```
test/me2e/
├── builder/                 # MLIR generation, compilation, and execution utilities
│   ├── thread_builder.py    # Base class with MLIR building blocks
│   ├── dm_builder.py        # Data movement thread builder (reader/writer)
│   ├── compute_builder.py   # Compute thread builder
│   ├── ttl_builder.py       # Build TTL modules via Python bindings
│   ├── pipeline.py          # Pass pipeline execution
│   ├── kernels.py           # Kernel translation and metadata extraction
│   ├── ttnn_runner.py       # TTNN device execution (uses shared kernel_runner)
│   ├── dtype_utils.py       # Dtype conversion utilities
│   └── device_arch.py       # Device architecture utilities
├── ops/                     # Operation tests
│   ├── __init__.py          # Auto-generation logic, base classes, op mappings
│   ├── test_simple.py       # MLIR builder validation tests
│   ├── test_binary.py       # Auto-generated binary op test classes
│   ├── test_unary.py        # Auto-generated unary op test classes
│   └── test_fused.py        # Custom fused operation tests
├── base.py                  # ME2ETestBase defining pipeline stages
├── runner.py                # Declarative test runner (executes full pipeline)
├── test_compute_ops.py      # Declarative parametrized tests (main test suite)
├── op_specs.py              # Operation specifications (auto-generated from .def file)
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
    buffer_factor: int = 2                 # 1=single, 2=double buffer (default)
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED
    buffer_type: BufferType = BufferType.DRAM
```

### Memory Configuration

The `memory_layout` and `buffer_type` fields control MLIR layout attribute generation:

**Buffer Types** (`BufferType` enum):
- `DRAM` (default): DRAM buffers
- `L1`: L1 memory buffers

**Memory Layouts** (`MemoryLayout` enum):
- `INTERLEAVED` (default): Standard interleaved layout
- `HEIGHT_SHARDED`: Height-sharded layout
- `WIDTH_SHARDED`: Width-sharded layout
- `BLOCK_SHARDED`: Block-sharded layout

**Examples**:
```python
# Default: DRAM + interleaved
config = E2EConfig(grid_shape=(2, 2))

# L1 buffers
config = E2EConfig(grid_shape=(2, 2), buffer_type=BufferType.L1)

# Height-sharded layout
config = E2EConfig(grid_shape=(2, 2), memory_layout=MemoryLayout.HEIGHT_SHARDED)

# L1 + block-sharded
config = E2EConfig(
    grid_shape=(2, 2),
    buffer_type=BufferType.L1,
    memory_layout=MemoryLayout.BLOCK_SHARDED
)
```

The builder automatically generates appropriate MLIR layout attributes based on these settings.

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
