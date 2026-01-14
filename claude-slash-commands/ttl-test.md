---
description: Generate test cases and a test runner for a TT-Lang kernel
argument-hint: <kernel-file> [test-description]
---

## Task

Create comprehensive tests for a TT-Lang kernel including entry point setup, test inputs, and validation.

## Input

$ARGUMENTS

## Process

1. **Analyze the kernel**
   - Parse the kernel signature and parameters
   - Identify input/output tensor shapes and types
   - Understand the computation being performed
   - Note any constraints or preconditions

2. **Design test cases**

   Create a comprehensive test suite covering:
   - **Basic functionality**: Simple inputs that exercise the main path
   - **Edge cases**: Empty tensors, single elements, boundary sizes
   - **Shape variations**: Different tile counts, aspect ratios
   - **Data type coverage**: Test supported data types (f32, bf16, etc.)
   - **Numerical edge cases**: Zeros, negative numbers, large values, NaN/Inf handling

3. **Generate reference implementation**
   - Create a PyTorch/NumPy reference for expected outputs
   - Ensure reference handles all test cases correctly

4. **Create test entry point**
   - Write the TTNN integration code
   - Set up device and memory allocation
   - Handle data transfer to/from device

5. **Build test runner**
   - Create pytest-compatible test functions
   - Add proper assertions with tolerance for floating point
   - Include timing and performance validation if relevant
   - Add descriptive test names and docstrings

6. **Generate test inputs**
   - Create deterministic test data with fixed seeds
   - Include both random and structured inputs
   - Ensure reproducibility across runs

## Output

Provide:
- Test file with all test cases
- Reference implementation for validation
- Entry point code for TTNN integration
- Instructions for running the tests
- Example test output showing expected results
