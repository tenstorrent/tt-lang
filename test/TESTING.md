# Python Lit Testing Guide

## Overview

Python lit tests verify DSL functionality by checking generated MLIR at two stages:
- **Initial IR**: Before compilation pipeline (d2m.generic, linalg ops, streams)
- **Lowered IR**: After compilation pipeline (ttmetal ops, kernel functions, emitc calls)

## Test Structure

Each test generates two MLIR files automatically:
- `%t.initial.mlir` - IR before pipeline execution
- `%t.final.mlir` - IR after full compilation pipeline

**Important:** Each test file should contain only one `@pykernel_gen` decorated function. Multiple kernels in one file will overwrite the temp files, causing only the last kernel to be checked.

## RUN Commands

**Line 1:** Execute test and generate temp files
```python
# RUN: %python %s
```

**Line 2:** Check initial IR
```python
# RUN: FileCheck %s < %t.initial.mlir
```

**Line 3:** Check lowered IR
```python
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
```

## Writing Good Checks

### Initial IR: Thorough Verification

Check the complete structure using SSA value captures.

**Pattern:**
```python
# CHECK-LABEL: func.func @test_name
# CHECK-SAME: (%[[ARG0:.+]]: tensor<{{.*}}> {d2m.stream = false}
# CHECK-SAME:  %[[ARG1:.+]]: tensor<{{.*}}> {d2m.stream = false})

# Verify operation structure
# CHECK: %[[RESULT:.+]] = d2m.some_op
# CHECK-SAME: attribute = value
# CHECK-SAME: ins(%[[ARG0]]
# CHECK-SAME: outs(%[[ARG1]]

# Verify region arguments
# CHECK: ^region0(%[[CB0:.+]]: !d2m.cb<tensor<{{.*}}>
# CHECK-SAME: %[[CB1:.+]]: !d2m.cb<tensor<{{.*}}>):

# Verify operations use captured values
# CHECK: %[[VAL:.+]] = d2m.wait %[[CB0]]
# CHECK: d2m.store %[[CB1]], %[[VAL]]

# CHECK: return %[[RESULT]]
```

**Key practices:**
- Capture SSA values: `%[[NAME:.+]]`
- Reuse captures to verify data flow
- Use `{{.*}}` for attributes that may vary
- Use `{{[0-9]+}}` for generated numeric suffixes
- Check types: `!d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>`
- Verify attributes: `grid = #ttcore.grid<2x2>`

### Lowered IR: Smoke Testing

Verify core functionality without exhaustive checks.

**Pattern:**
```python
# CHECK-LOWERED-LABEL: func.func @test_name

# Verify: Lowered to ttmetal
# CHECK-LOWERED: ttmetal.enqueue_program

# Verify: Kernel functions generated
# CHECK-LOWERED: func.func private @datamovement_kernel
# CHECK-LOWERED: func.func private @compute_kernel

# Verify: Core compute operation present
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"
```

**Focus on:**
- Top-level structure (ttmetal.enqueue_program)
- Kernel functions exist
- Key compute operations (tile_add, matmul, etc.)

**Skip:**
- Exact SSA value names
- Register allocation details
- Intermediate operations

## Running Tests

```bash
cd /path/to/tt-lang
source env/activate
export SYSTEM_DESC_PATH=$(pwd)/system_desc.ttsys

# Run all Python tests
llvm-lit -sv test/python/

# Run single test
llvm-lit -sv test/python/test_generic_op.py
```

## Debugging Failed Tests

Generated MLIR files are saved in `test/temp/python/Output/`:

```bash
# View initial IR
cat test/temp/python/Output/test_generic_op.py.tmp.initial.mlir

# View lowered IR
cat test/temp/python/Output/test_generic_op.py.tmp.final.mlir
```

Compare CHECK patterns against actual output to identify mismatches.
