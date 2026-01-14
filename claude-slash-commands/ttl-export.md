---
description: Export TT-Lang kernel to TT-Metal C++ kernel code
argument-hint: <kernel-file>
---

## Task

Compile a TT-Lang kernel and generate clean, readable TT-Metal C++ code.

## Input

$ARGUMENTS

## Process

1. **Compile the TT-Lang kernel**
   - Run the compiler pipeline on the kernel
   - Capture the generated C++ output
   - Note any compilation warnings or errors

2. **Extract the generated C++ code**
   - Locate the EmitC output from the compiler
   - Identify the data movement and compute kernel sections

3. **Beautify the C++ code**
   - Remove unnecessary casts and type annotations
   - Improve variable names for readability:
     - Replace generated names with semantic names
     - Use descriptive names for circular buffers
     - Name loop variables appropriately
   - Format code according to TT-Metal style guidelines
   - Add comments explaining key sections

4. **Validate the output**
   - Ensure the beautified code compiles
   - Verify semantic equivalence with the original

## Output

Provide:
- The cleaned-up TT-Metal C++ kernel code
- Separate files for data movement and compute kernels if applicable
- Integration instructions for using in TT-Metal/TT-NN
