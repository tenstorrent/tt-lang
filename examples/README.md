# Examples

This directory contains working examples demonstrating tt-lang capabilities for authoring custom kernels on Tenstorrent hardware.

## Available Examples

### Core Examples

- **custom_dm_matmul.py** - Demonstrates custom data movement with matrix multiplication operations, showcasing how to use the `@pykernel_gen` decorator for complex kernels
- **eltwise_add.py** - Element-wise addition kernel showing basic tile operations
- **test_simple_add.py** - Simple addition kernel demonstrating minimal kernel structure
- **test_accessor_creation.py** - Examples of TensorAccessor usage patterns for indexed tile-level access

### Utilities

- **utils.py** - Shared utility functions used across examples

## Running Examples

### Prerequisites

1. Build tt-lang following the instructions in the main [README.md](../README.md)
2. Activate the environment:
   ```bash
   cd /path/to/tt-lang
   source build/env/activate
   ```

3. Set up system descriptor (required for some examples):
   ```bash
   export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys
   ```

   > **Note:** The system descriptor file is not included in the repository. Contact your Tenstorrent representative for access to hardware configuration files.

### Running an Example

```bash
# Basic execution
python examples/custom_dm_matmul.py

# With MLIR output for debugging
export TTLANG_INITIAL_MLIR=/tmp/initial.mlir
export TTLANG_FINAL_MLIR=/tmp/final.mlir
python examples/custom_dm_matmul.py

# View generated MLIR
cat /tmp/initial.mlir
cat /tmp/final.mlir
```

## Understanding the Examples

Each example demonstrates different aspects of the tt-lang DSL:

- **Data Movement**: How to transfer data between memory spaces (DRAM ↔ L1)
- **Tile Operations**: Working with tile-level compute operations
- **Memory Management**: Using CircularBuffers and managing L1/DRAM allocation
- **Synchronization**: Multi-core coordination with semaphores

For comprehensive documentation on the DSL and its features, see the [Hitchhiker's Guide](../docs/HITCHHIKERS_GUIDE.md).

## Debugging Examples

If an example fails, you can enable verbose pass output to trace the compilation pipeline:

```bash
export TTLANG_VERBOSE_PASSES=1
python examples/custom_dm_matmul.py 2>&1 | tee /tmp/pipeline.log

# Search for specific operations or errors
grep -i "error\|warning" /tmp/pipeline.log
```

For more debugging techniques, refer to the [Testing Guide](../test/TESTING.md).
