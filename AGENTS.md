# TT-Lang Agent Guidelines

## Build/Lint/Test Commands
- **Environment**: `source build/env/activate` (activate virtual environment
  first, use actual build dir)
- **Configure**: with internal tt-mlir build `cmake -G Ninja -B build`; with
  pre-build tt-mlir
  `cmake -G Ninja -B build -DTTMLIR_BUILD_DIR=/path/to/tt-mlir/build`; with
  pre-installed tt-mlir
  `cmake -G Ninja -B build -DTTMLIR_DIR=/path/to/tt-mlir/build/lib/cmake/ttmlir`
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
- Be precise and concise. Avoid replicating code, refactor redundant implementations.
- **C++ Style**: LLVM style (see .clang-format, .clang-tidy)
- **Naming**: UpperCamelCase for types, lowerCamelCase for variables/functions
- **Includes**: Absolute paths from tt-lang root, sorted: main header → local →
  LLVM → system
- **Comments**: Full sentences, explain why not what, TODO with alias and issue
  link
- **Python**: PEP 8 with black formatter (v23.x), Python 3.10+ only
  namespace for .cpp
- **Namespaces**: Lowercase, avoid `using namespace`, no aliases in headers
- **Error Handling**: Early returns to reduce nesting, no alternative tokens (&&
  not and)

## MLIR implementation
- Follow the conventions in llvm-project for directory organization and naming
  conventions.
- **Dialect design**: Don’t recover semantic info by chasing SSA, encode it in
  the operations/types/etc.
- **MLIR passes (modern pattern)**: Define passes in `Passes.td`; let TableGen
  emit factories/registration. In the `.cpp`, include `Passes.h.inc` with
  `GEN_PASS_DEF_...`, derive from the generated `...Base`, implement
  `runOnOperation()`, and rely on the generated `create*Pass()` (no manual
  constructors).
- **Transforms layout**: Dialect-specific pass definitions in
  `include/ttlang/<Dialect>/Passes.td`, headers in
  `include/ttlang/Dialect/<Dialect>/{IR,Transforms,TransformOps,Utils}` and
  implementations in `lib/Dialect/<Dialect>/{IR,Transforms,TransformOps,Utils}`.
- **Pass naming and deps**: Prefix pass names with the dialect acronym (e.g.,
  `TTLConvert...`). In `dependentDialects`, list only dialects for ops the pass
  creates; do not include the starting dialect.
- **Debugging**: use `--debug-only=dialect-conversion` with `ttlang-opt`
- Use enums instead of integer literals for encoding items in a category.

### Pattern Rewriter Error Handling
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

### Lit tests
- Always add a brief comment in front of tests to specify the purpose of the
  test. Add a concise summary on top of the test file about what is being
  tested.
- Use `--split-input-file` for multiple lit tests in the same file.
- Use "// PREFIX-NEXT:" (for FileCheck prefix PREFIX) whenever possible instead of just "// PREFIX:"
- **Negative/invalid tests**: should be in a file named *_invalid.<suffix>. For
  invalid tests, use `--verify-diagnostics` and `expected-error @below` as well
  as `--split-input-file` if file contains multiple tests.
- **CHECK-LABEL**: Start each test function
- **CHECK-NEXT**: Verify operation ordering (catches extra/missing ops)
- **Capture variables**: `%[[VAR:.*]]` for reuse in subsequent checks
- **Verify data flow**: Check that operations consume correct SSA values
- **CHECK-NOT**: Ensure unwanted operations/attributes are not present

## When Uncertain
- Ask clarifying questions rather than assuming
- If multiple valid MLIR patterns exist, present tradeoffs before implementing
- Flag when a request conflicts with these standards

## Documentation Style
- **Tone**: Formal and technical; avoid second person ("you/your")
- **Voice**: Use present tense descriptive style ("provides", "enables", "includes")
- **Structure**: Keep sentences clear and concise; end with periods
- **Content**: Explain what and why; avoid unnecessary fluff
- **Code examples**: Include complete, runnable examples where appropriate
- **References**: Follow LLVM documentation style: https://llvm.org/docs/

## Additional Notes
- **Agent Design Principle**: Implement only the minimum necessary
  functionality; avoid feature creep and arbitrary expansions
- **PR Descriptions**: Use this template: .github/pull_request_template.md
- Use `pre-commit run --all-files` before commits
- Prefer `git mv` to deleting and adding files that are in git. Stop and ask
  user to do if you can't do it.
- Generate commit messages and PR summaries in plain ASCII format using github
  markdown. When appropriate, include plain ASCII diagrams.
- Follow LLVM coding standards: https://llvm.org/docs/CodingStandards.html
- Follow best practices: https://llvm.org/docs/ProgrammersManual.html
