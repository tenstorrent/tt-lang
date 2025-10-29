# tt-lang Project Guide for Claude

## Project Overview

tt-lang is the DSL (Domain Specific Language) and Python API layer for tt-mlir. It provides the `@pykernel_gen` decorator for writing custom data movement and compute kernels in Python, along with D2M dialect Python bindings.

tt-lang depends on tt-mlir and reuses its entire toolchain (LLVM, MLIR, Python environment).

## Required Reading

**You MUST read these before working on tt-lang**:
- `docs/HITCHHIKERS_GUIDE.md` - Complete DSL guide, pipeline architecture, shapes, passes, examples
- `docs/BUILD_SYSTEM.md` - Build system architecture and integration with tt-mlir
- `docs/TESTING.md` - Python lit testing guide for D2M dialect

**For macOS builds**:
- `../tt-mlir/MACOS_BUILD.md` - macOS-specific build instructions (supersedes README build instructions on macOS)

Detect the OS you're running on and reference the appropriate build documentation.

## Environment Setup and Building

### System Descriptor File

A `.ttsys` file is required for running SOME examples and tests (describes hardware configuration).

**IMPORTANT**: The system descriptor is NOT in the repository. When you need one for running examples/tests, **ask the user** to provide the path. Do not search for it in the repo or common locations - let the user tell you where it is. Only prompt the user if you actually need this.

### Detecting Repository Layout

Before building, determine where tt-lang and tt-mlir are located relative to each other:
```bash
# From tt-lang directory
pwd                          # Get tt-lang path
ls -d ../tt-mlir 2>/dev/null # Check if tt-mlir is sibling directory
echo $TT_MLIR_HOME          # Check if user has set this
```

If tt-mlir is not in the default sibling location, the user will need to set `TT_MLIR_HOME` before activating the environment.

### Building tt-mlir (Prerequisite)

**tt-lang requires a working tt-mlir build.** You must ensure tt-mlir is built before building tt-lang:

```bash
cd <tt-mlir-directory>
source env/activate

# Configure and build tt-mlir
# Follow instructions in MACOS_BUILD.md (macOS) or README.md (Linux)
cmake -G Ninja -B build <options>
cmake --build build
```

**tt-mlir Build Issues**: tt-mlir builds can be non-deterministic, specifically around **nanobind** Python bindings. If you see:
- Compiler errors mentioning `nanobind`
- Something that seems totally unrelated to your change

Then retry the build 2-3 times. This is a known issue with nanobind and parallel builds.

### Activating tt-lang Environment

```bash
# Navigate to tt-lang root
cd <tt-lang-directory>

# If tt-mlir is not at ../tt-mlir, set TT_MLIR_HOME first
export TT_MLIR_HOME=/path/to/tt-mlir  # Optional, only if not sibling dir

# Activate environment (automatically sources tt-mlir's environment)
source env/activate

# Only set when running examples/tests - ask user for path when needed
# export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys

# Verify activation
echo $TTLANG_ENV_ACTIVATED  # Should be 1
echo $TT_LANG_HOME         # Should be tt-lang root
echo $TT_MLIR_HOME         # Should be tt-mlir root
```

### Building tt-lang

See `docs/BUILD_SYSTEM.md` for detailed build instructions and options.

```bash
cd <tt-lang-directory>
source env/activate

# Configure
cmake -G Ninja -B build .

# Build
cmake --build build

# Rebuild after code changes
cmake --build build
```

tt-lang builds are generally reliable. If builds fail, check that tt-mlir built successfully first.

## Testing

See `docs/TESTING.md` for complete testing documentation including:
- Running lit tests
- Test output locations
- Writing new tests
- FileCheck patterns

**Quick reference**:
```bash
cd <tt-lang-directory>
source env/activate
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys  # Ask user for path

# Run all tests
llvm-lit -sv test/python/

# Run example
python example/custom_dm_matmul.py # optionally specify initial and final MLIR dump locations
```

## MLIR Generation and Debugging

### Saving MLIR to Files

```bash
# Save initial and final MLIR
export TTLANG_INITIAL_MLIR=/tmp/test_initial.mlir
export TTLANG_FINAL_MLIR=/tmp/test_final.mlir

python examples/custom_dm_matmul.py

# View the files
cat /tmp/test_initial.mlir
cat /tmp/test_final.mlir
```

### Observing Pass Pipeline

Enable verbose pass output to see IR after each pass:
```bash
export TTLANG_VERBOSE_PASSES=1
export TTLANG_INITIAL_MLIR=/tmp/verbose_initial.mlir
export TTLANG_FINAL_MLIR=/tmp/verbose_final.mlir

python examples/custom_dm_matmul.py 2>&1 | tee /tmp/full_pipeline.log
```

This produces ~50 pass IR dumps showing transformations. The output is VERY LARGE (thousands of lines). Use grep/tail to navigate:
```bash
# See which passes ran
grep "^// -----// IR Dump" /tmp/full_pipeline.log

# Extract IR for a specific pass
grep -A 200 "Before D2MAllocate" /tmp/full_pipeline.log
```

### Pass Pipeline Configuration

Located in `python/ttlang/d2m_api.py` lines 302-332:
- `TTLANG_VERBOSE_PASSES` - Enable verbose output (line 322)
- `TTLANG_INITIAL_MLIR` - Save initial IR (line 296)
- `TTLANG_FINAL_MLIR` - Save final IR (line 334)
- `use_tile_matmul` - Use tile-level matmul (line 304, currently True)
- Pipeline string (line 305): `d2m-generic-replace-globals,ttir-to-ttmetal-pipeline{use-tile-matmul=1}`

Key passes in order:
1. `ttcore-register-device` - Register device configuration
2. `d2m-generic-replace-globals` - Replace globals with function args
3. Frontend passes - Fusion, canonicalization
4. `one-shot-bufferize` - Tensor to memref conversion
5. `d2m-bufferize-function-args` - Restore layout attrs
6. `d2m-allocate` - Assign L1/DRAM addresses
7. `d2m-generic-lower-dmas` - Lower DMA to hardware ops
8. Backend passes - D2M→TTKernel→EmitC

## Debugging Workflows Using Sub-Agents

### Workflow 1: Trace Issue Through Pass Pipeline

**When to use**: Compilation error, assertion failure, or unexpected behavior in pipeline.

**If the user asks you to trace the MLIR pass pipeline for an error**, follow this workflow and provide a response structured like the example below.

**Approach**:
1. Run with verbose passes to capture all IR
2. Identify the failing pass from error message
3. Use an **Explore agent** to search the verbose output for the error and extract relevant passes
4. Use a **general-purpose agent** to analyze the IR transformation before/after the problem
5. Provide a succinct summary of each pass where important ops change, are converted, or transformed
6. Provide context around important operands and instructions
7. Highlight where these change throughout the passes
8. Highlight the error
9. Speculate on what might have caused it if possible

**Example Response Format**:

Notice how in the example below there are snippets of MLIR, several key passes that transform or convert MLIR, and relevant context (such as ops outside of the block for context). Also the error clearly highlighted at the end. The snippets with a few context operations around them are EXTRMELY HELPFUL.

---

I traced the compilation error through the pass pipeline. Here's what I found:

**Error**: Type mismatch in `D2MInsertDstRegisterAccess` pass

**Pass-by-Pass Analysis**:

**Pass 1: Before TTCoreRegisterDevicePass**
```mlir
func.func @test_sign_f32(%arg0: tensor<128x96xf32>, %arg1: tensor<128x96xf32>) -> tensor<128x96xf32> {
  %0 = "ttir.sign"(%arg0, %arg1) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
  return %0 : tensor<128x96xf32>
}
```
- High-level TTIR operation: `ttir.sign` on f32 tensors
- Input/output types are all `tensor<128x96xf32>`

**Pass 2: Before OneShotBufferizePass**
```mlir
^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
  %8 = "d2m.tile_typecast"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
  %9 = "d2m.tile_sign"(%8) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  %10 = "d2m.tile_typecast"(%9) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f32>
  linalg.yield %10 : !ttcore.tile<32x32, f32>
```
- **Key transformation**: TTIR lowered to D2M tile operations
- Sign operation converted to work with f16 (hardware requirement)
- Typecasts inserted: f32 → f16 → sign → f16 → f32
- Block arguments and yield are both f32 tiles

**Pass 3: Before D2MInsertDstRegisterAccess**
```mlir
linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%cb0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
    outs(%cb1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
  %0 = "d2m.tile_typecast"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
  %1 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
  %2 = "d2m.tile_typecast"(%1) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f32>
  linalg.yield %2 : !ttcore.tile<32x32, f32>
}
```
- **Bufferization complete**: tensors → memrefs with L1 memory space
- Input/output memrefs are f32: `memref<1x1x!ttcore.tile<32x32, f32>, #l1>`
- Block still performs f32→f16→f32 typecast sequence
- Everything type-checks correctly at this stage

**Pass 4: After D2MInsertDstRegisterAccess (FAILED)**
```mlir
%4 = "d2m.acquire_dst"() : () -> memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>

%6 = "affine.load"(%4, %arg6, %arg7) : (memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>, index, index) -> !ttcore.tile<32x32, f32>
%7 = "d2m.tile_typecast"(%6) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
"affine.store"(%7, %4, %3, %arg6, %arg7) : (!ttcore.tile<32x32, f16>, memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>, index, index, index) -> ()
//                                          ^^^^^^^^^^^^^^^^^^^^^ f16 value
//                                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ f32 memref

// ERROR:
error: 'affine.store' op value to store must have the same type as memref element type
note: "affine.store"(%7, %4, %3, %arg6, %arg7) :
      (!ttcore.tile<32x32, f16>, memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>, ...) -> ()
      ^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      f16 tile                   f32 memref - TYPE MISMATCH
```

**Root Cause**: The `D2MInsertDstRegisterAccess` pass inserted destination register operations (acquire_dst, affine.load, affine.store) but failed to account for intermediate typecast operations. It stored an f16 tile into an f32 memref.

**What went wrong**:
1. Pass acquired dst register as f32: `memref<4x1x1x!ttcore.tile<32x32, f32>, #dst>`
2. Pass loaded f32 tile from dst register
3. Pass applied typecast to f16
4. Pass tried to store f16 result back to f32 dst register → TYPE MISMATCH

---

**Key commands for manual debugging**:
```bash
# Find which pass failed
grep -i "error\|failed\|assertion" /tmp/pipeline_error.log

# Extract IR before specific pass
grep -B 5 "Before PASSNAME" /tmp/pipeline_error.log | grep "^//"
grep -A 200 "Before PASSNAME" /tmp/pipeline_error.log > /tmp/before_pass.mlir

# Extract IR after specific pass (or before next pass)
grep -A 200 "Before NEXT_PASS" /tmp/pipeline_error.log > /tmp/after_pass.mlir
```

### Workflow 2: Highlight Relevant Snippets for Single Op/Error

**When to use**: Tracking how a specific operation transforms, or finding all instances of an error pattern.

**Approach**:
1. Identify the operation/pattern to track (e.g., `d2m.dma`, `tile_matmul`, or error message)
2. Use an **Explore agent** to search through MLIR dumps and extract all occurrences
3. Show transformation progression through the pipeline

**Example workflow**:
```bash
# Step 1: Generate verbose pipeline output
export TTLANG_VERBOSE_PASSES=1
python examples/test.py 2>&1 > /tmp/pipeline.log

# Step 2: Launch Explore agent to find all occurrences
# Agent task: "Search /tmp/pipeline.log for all occurrences of 'd2m.dma'
#              operations. For each occurrence, extract 5 lines before and
#              10 lines after. Group by pass name. Show how the operation
#              signature changes through the pipeline."

# Step 3: Analyze the transformation pattern
# Look for: type changes, attribute additions, operation replacements
```

**Key commands for manual debugging**:
```bash
# Find all occurrences with context
grep -B 5 -A 10 "d2m.dma" /tmp/pipeline.log

# Find in specific passes only
grep -A 200 "Before D2MAllocate" /tmp/pipeline.log | grep "d2m.dma"

# Count occurrences per pass
grep "^// -----// IR Dump" /tmp/pipeline.log > /tmp/passes.txt
grep -n "d2m.dma" /tmp/pipeline.log > /tmp/ops.txt
# Then compare line numbers to see which passes have the op
```

### Workflow 3: Compare MLIR Dumps Through Pipeline

**When to use**:
- **Comparing across branches**: Build one branch and save output, build another and compare
- **Comparing similar tests**: One test fails, one succeeds - compare their MLIR to identify differences
- **Testing targeted compiler changes**: Make a change to a pass, compare MLIR before/after to verify impact

**Approach**:
1. Generate MLIR dumps for both scenarios (branches, tests, or before/after change)
2. Use a **general-purpose agent** to run diffs and analyze changes
3. Identify what ops were added, removed, or modified

**Example workflow for branch comparison**:
```bash
# Save MLIR from branch A
git checkout branch-a
cmake --build build
export TTLANG_INITIAL_MLIR=/tmp/branch_a_initial.mlir
export TTLANG_FINAL_MLIR=/tmp/branch_a_final.mlir
python examples/test.py

# Save MLIR from branch B
git checkout branch-b
cmake --build build
export TTLANG_INITIAL_MLIR=/tmp/branch_b_initial.mlir
export TTLANG_FINAL_MLIR=/tmp/branch_b_final.mlir
python examples/test.py

# Launch general-purpose agent to compare
# Agent task: "Compare MLIR outputs from two branches:
#              - branch A: /tmp/branch_a_final.mlir
#              - branch B: /tmp/branch_b_final.mlir
#              Summarize what changed between branches."
```

**Example workflow for test comparison**:
```bash
# Run passing test
export TTLANG_FINAL_MLIR=/tmp/passing_test.mlir
python test/passing_example.py

# Run failing test
export TTLANG_FINAL_MLIR=/tmp/failing_test.mlir
python test/failing_example.py

# Compare to find what's different
```

**Key commands for manual debugging**:
```bash
# Basic diff
diff /tmp/compare_initial.mlir /tmp/compare_final.mlir

# Count operations by type
grep -o "[a-z0-9_]*\.[a-z0-9_]*" /tmp/compare_initial.mlir | sort | uniq -c
grep -o "[a-z0-9_]*\.[a-z0-9_]*" /tmp/compare_final.mlir | sort | uniq -c

# Extract specific function for comparison
sed -n '/func.func @matmul/,/^  }/p' /tmp/compare_initial.mlir > /tmp/func_initial.mlir
sed -n '/func.func @matmul/,/^  }/p' /tmp/compare_final.mlir > /tmp/func_final.mlir
diff /tmp/func_initial.mlir /tmp/func_final.mlir
```

**Advanced: Extract intermediate checkpoints**:
```bash
# Run with verbose passes
export TTLANG_VERBOSE_PASSES=1
python examples/test.py 2>&1 > /tmp/full_pipeline.log

# Extract IR after bufferization
sed -n '/After OneShotBufferizePass/,/^\/\/ -----\/\/ IR Dump Before/p' /tmp/full_pipeline.log > /tmp/after_bufferize.mlir

# Extract IR after allocate
sed -n '/After D2MAllocate/,/^\/\/ -----\/\/ IR Dump Before/p' /tmp/full_pipeline.log > /tmp/after_allocate.mlir

# Now compare initial → after_bufferize → after_allocate → final
```

## GitHub Issues

View and work with issues:
```bash
cd <tt-lang-directory>
gh issue list --limit 20
gh issue view NUMBER
```

## Key Files and Locations

### Related tt-mlir Files
- `../tt-mlir/lib/Dialect/D2M/` - D2M dialect implementation
- `../tt-mlir/lib/Dialect/TTKernel/` - TTKernel dialect
- `../tt-mlir/lib/Conversion/` - Conversion passes (D2M→TTKernel, etc)
- `../tt-mlir/test/ttmlir/Dialect/D2M/` - MLIR lit tests for D2M dialect

## Tips for Claude

### Environment Management
- Always source `env/activate` before running commands
- Only set `SYSTEM_DESC_PATH` when running examples/tests (ask user for path when needed)
- Check `$TTLANG_ENV_ACTIVATED` to verify environment is active
- Detect OS and reference appropriate build docs (MACOS_BUILD.md for macOS)

### Build Issues
- **tt-mlir** builds can fail randomly due to **nanobind** - retry 2-3 times if you see nanobind/Python binding errors
- **tt-lang** builds are generally reliable - if they fail, check tt-mlir built successfully first
- If CMake can't find tt-mlir, check `$TT_MLIR_HOME` or verify tt-mlir is in sibling directory

### Test Output Management
- Test output is VERY LARGE - use `tail -200` to see relevant parts
- Save verbose pipeline output to files, don't dump in terminal
- Use grep to filter for specific passes or operations

### MLIR Analysis
- Use sub-agents (Explore, general-purpose) for large MLIR analysis tasks
- Extract specific functions/regions rather than analyzing entire modules
- Count operations to get high-level view before detailed analysis
- Follow the Workflow 1 example format when tracing errors through pipeline

### Error Investigation
- Error messages often include pass names - use those to find relevant IR
- Look for "Before PASSNAME" in verbose output to get pre-error state
- Compare IR before and after the failing pass to identify the problem

### When to Ask User
- When SYSTEM_DESC_PATH is needed for running examples/tests
- If example/test is unclear which to run
- If multiple possible approaches exist for solving a problem
- If error is ambiguous and user context would help
