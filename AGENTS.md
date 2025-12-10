# TT-Lang Agent Guidelines

## Build/Lint/Test Commands
- **Environment**: `source env/activate` (activate virtual environment first)
- **Configure**: with internal tt-mlir build `cmake -G Ninja -B build`;
with pre-build tt-mlir `cmake -G Ninja -B build -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build`;
with pre-installed tt-mlir `cmake -G Ninja -B build -DTTMLIR_DIR=/path/to/tt-mlir/build/lib/cmake/ttmlir`
- **Build**: `cmake --build build`
- **Lint**: `pre-commit run --all-files` (includes clang-format, black,
  copyright checks)
- **Compiler tests**: `cmake --build build --target check-ttlang`
- **Single MLIR test**: `llvm-lit test/ttlang/path/to/test.mlir`
- **Python tests**: `pytest test/python` (requires `ttrt query --save-artifacts`
  and `SYSTEM_DESC_PATH`)
- **Runtime tests**: `llvm-lit test/python/` (hardware execution tests, require
  `SYSTEM_DESC_PATH`)
- **Simulation tests**: `pytest test/sim/` (software simulation of runtime
  behavior)

## Code Style Guidelines
- **C++ Style**: LLVM style (see .clang-format, .clang-tidy)
- **Naming**: UpperCamelCase for types, lowerCamelCase for variables/functions
- **Includes**: Absolute paths from tt-lang root, sorted: main header → local →
  LLVM → system
- **Comments**: Full sentences, explain why not what, TODO with alias and issue
  link
- **Python**: PEP 8 with black formatter (v23.x), Python 3.10+ only
- **Functions**: Bottom-up order, helpers before callers, static/anonymous
  namespace for .cpp
- **Namespaces**: Lowercase, avoid `using namespace`, no aliases in headers
- **Error Handling**: Early returns to reduce nesting, no alternative tokens (&&
  not and)

## Pattern Rewriter Error Handling
- **NEVER call `emitOpError()` inside a pattern rewriter** - causes pass to
  succeed while emitting diagnostics
- Inside patterns: Use `rewriter.notifyMatchFailure()` for pattern match
  failures
- In `runOnOperation()`: Use `op.emitOpError()` + `signalPassFailure()` for
  precondition checks
- Why: `emitOpError()` in a pattern returns pattern failure (not pass failure),
  greedy rewriter continues, pass succeeds with diagnostics, downstream crashes
  occur (e.g., pytest failing with
  `mlir::python::PyMlirContext::ErrorCapture::~ErrorCapture(): Assertion `errors.empty()
  && "unhandled captured errors"' failed.`)

## Additional Notes
- **Agent Design Principle**: Implement only the minimum necessary
  functionality; avoid feature creep and arbitrary expansions
- **PR Descriptions**: Use this template:
  ```
  ### Problem description
  [Explain the issue and why this change is needed]

  ### What's changed
  [Describe what was actually modified, focusing on rationale and design decisions]

  ### Checklist
  - [ ] New/Existing tests provide coverage for changes
  ```
- Use `pre-commit run --all-files` before commits
- Create GitHub issues for TODOs with format:
  `TODO (alias): description. Issue: #123`
- Follow LLVM coding standards: https://llvm.org/docs/CodingStandards.html
- Follow best practices: https://llvm.org/docs/ProgrammersManual.html
