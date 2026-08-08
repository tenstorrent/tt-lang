# TT-Lang Agent Guidelines

## Build/Lint/Test Commands
- **Environment**: `source build/env/activate` (activate virtual environment first, use actual build dir)
- **Configure**: `cmake -G Ninja -B build`;
  with pre-built LLVM: `cmake -G Ninja -B build -DMLIR_PREFIX=/path/to/llvm-install`;
  with tt-lang toolchain: `cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON`
- **Build**: `cmake --build build`
- **Lint**: `pre-commit run --all-files` (includes clang-format, black,
  copyright checks)
- **Compiler tests**: `cmake --build build --target check-ttlang`
- **Single MLIR test**: `llvm-lit test/ttlang/path/to/test.mlir`
- **ME2E tests**: `pytest test/me2e/`(requires ttnn and a TT device)
- **Pytest tests**: `pytest test/python` (requires ttnn and a TT device)
- **Python lit tests**: `llvm-lit test/python/` (hardware execution tests)
- **Simulation tests**: `pytest test/sim/` (software simulation of runtime behavior); add `--run-slow` to include slow tests (hardware CI always passes this flag; GitHub-hosted CI does not)

## Docker

- Run Docker commands directly. Never use `sudo docker`; the user has Docker
  daemon access through group membership.

## Code Style Guidelines
- **C++ Style**: LLVM style (see .clang-format, .clang-tidy)
- **Naming**: UpperCamelCase for types, lowerCamelCase for variables/functions
- **Includes**: Absolute paths from tt-lang root, sorted: main header → local →
  LLVM → system
- **Comments**: See [Comments](#comments) below.
- **Python**: PEP 8 with black formatter (v23.x), Python 3.10+ only
- **Functions**: Bottom-up order, helpers before callers, static/anonymous
  namespace for .cpp
- **Namespaces**: Lowercase, avoid `using namespace`, no aliases in headers
- **Error Handling**: Early returns to reduce nesting, no alternative tokens (&&
  not and)
- **Callback Naming**: Name callbacks by the computation they perform, following
  upstream MLIR conventions such as `computeUbMinusLb`, not by when they are
  consulted. Use `ValueEvaluator` / `valueEvaluator` for integer value
  evaluation callbacks; do not call them fallbacks.
- **Unicode**: Avoid Unicode characters in code and documentation. Use ASCII
  equivalents instead (e.g., `->` instead of `→`). This ensures compatibility
  across different editors, terminals, and build environments.

### Comments

A comment exists for one reason: to let the next reader understand the code.
Comments occupy space and must be maintained, so write one only when the code
itself cannot carry the information.

- **Follow LLVM commenting conventions** in substance, location, and style:
  https://llvm.org/docs/CodingStandards.html#commenting. Use `///` doxygen
  comments on declarations in headers, `//` for implementation notes in `.cpp`
  files, and write full sentences with capitalization and a period. Do not
  repeat a header's doc comment above the definition.
- **Default to writing no comment.** The function name, signature, and body
  usually suffice. A comment is justified only when it states one of:
  1. Contract or role -- what callers can rely on, not the steps taken.
  2. Why this and not the obvious alternative -- the constraint, prior bug, or
     framework quirk that motivated the choice.
  3. An invariant the type system cannot express.
- **Do not narrate the code.** Reject comments that restate the function name,
  paraphrase the body in English, explain the *absence* of code, or hedge
  informally ("we just", "basically", "the trick is"). One line beats three.
- **Do not repeat a comment.** State a fact in exactly one place -- the
  declaration, the TableGen `description`, or a design document -- and nowhere
  else. Detailed op and pass documentation lives in the `.td`, not the `.cpp`;
  `.cpp` comments explain non-obvious implementation only.
- **Rationale belongs in the PR, not the source.** Design decisions,
  alternatives considered, performance measurements, and why a block of code
  sits in one location rather than another go in the PR description. When a
  decision must be recorded in the repository, write it up under `docs/`.
- **Prose rules match [Documentation Style](#documentation-style)**: no
  metaphorical verbs (write "execute", not "fire"; "delete", not "blow away"),
  no invented jargon, ASCII only.
- **TODOs** reference an issue number without an issue URL.

## MLIR implementation
- Follow the conventions in llvm-project for directory organization and naming
  conventions.
- **Dialect design**: Don’t recover semantic info by chasing SSA, encode it in the operations/types/etc.
- **MLIR passes (modern pattern)**: Define passes in `Passes.td`; let TableGen
  emit factories/registration. In the `.cpp`, include `Passes.h.inc` with
  `GEN_PASS_DEF_...`, derive from the generated `...Base`, implement
  `runOnOperation()`, and rely on the generated `create*Pass()` (no manual
  constructors).
- **Transforms layout**: Dialect-specific pass definitions in `include/ttlang/<Dialect>/Passes.td`,
  headers in `include/ttlang/Dialect/<Dialect>/{IR,Transforms,TransformOps,Utils}` and
  implementations in `lib/Dialect/<Dialect>/{IR,Transforms,TransformOps,Utils}`.
- **Pass naming and deps**: Prefix pass names with the dialect acronym
  (e.g., `TTLConvert...`). In `dependentDialects`, list only dialects for ops
  the pass creates; do not include the starting dialect.
- **Debugging**: use `--debug-only=dialect-conversion` with `ttlang-opt`
- Use enums instead of integer literals for encoding items in a category.

### Nonlocal Transformation Design

- When legality depends on multiple operations, regions, uses, or resource
  lifetimes, analyze the complete transformation scope on immutable IR before
  rewriting. Greedy pattern order must not determine semantic decisions.
- Record every application decision in a typed plan, including operands, uses,
  transactions, indexing, dependencies, and diagnostics. Application verifies
  and executes the plan; it does not rediscover policy. Discard the plan after
  any IR mutation.
- Model stateful resources with explicit acquisition, release, ownership, and
  publication transactions. Resource association, adjacency, use-list order,
  and integer identity are not lifetime proofs.
- Express repairs using semantic capabilities and exact consumer operands, not
  operation-specific exceptions. When planned materialization changes storage,
  distinguish the future input from current storage that constrains motion.
- Represent interacting candidates with explicit dependencies and
  preservation or erasure obligations. Compute mutually dependent requirements
  as a monotone fixed point over a finite set and canonicalize results by IR
  order.
- Reuse upstream dataflow, dead-code, dominance, region-order, purity,
  speculation, and operation-interface utilities for general compiler facts.
  Add custom analysis only for dialect-specific semantics. Unreachable and
  unknown are distinct; unknown results remain conservative.
- Use typed planned, rejected, and invalid-IR outcomes. Diagnose malformed IR
  and incomplete conversion before mutation; a rejected optimization leaves
  valid IR unchanged. Never leave partially rewritten IR after failure.
- Moving or recomputing operations across regions requires proof of SSA
  dominance, purity and speculation safety, absence of memory effects,
  resource availability at the destination, and preserved instrumentation
  ordering.
- When a transformation makes a new IR form common, audit downstream passes,
  verifiers, and every compiler pipeline. Implement the form or emit a precise
  diagnostic; do not rely on an assertion.
- Validate nonlocal transformations with systematic input matrices and
  baseline differential sweeps. Check generated-case counts and verifier
  filtering, isolate every failure cause, and cover all compiler modes,
  pipelines, dtypes, control-flow placements, and runtime-visible regressions.
- Document the algorithm, correctness argument, assumptions, conservative
  behavior, limitations, and upstream reuse in a design document.

### Op Creation API
- Use the static `OpTy::create(builder, loc, ...)` form, **not** the deprecated
  `builder.create<OpTy>(loc, ...)`. The latter is deprecated in current LLVM and
  will be removed.
  ```cpp
  // Good
  auto op = MyOp::create(rewriter, loc, resultType, operands);

  // Deprecated -- do not use
  auto op = rewriter.create<MyOp>(loc, resultType, operands);
  ```

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
- Always add a brief comment in front of tests to specify the purpose of the test. Add a concise summary on top of the test file about what is being tested.
- Use `--split-input-file` for multiple lit tests in the same file.
- **Negative/invalid tests**: should be in a file named *_invalid.<suffix>. For invalid tests, use `--verify-diagnostics` and `expected-error @below` as well as `--split-input-file` if file contains multiple tests.
- **CHECK-LABEL**: Start each test function
- **CHECK-NEXT**: Verify operation ordering (catches extra/missing ops)
- **Capture variables**: `%[[VAR:.*]]` for reuse in subsequent checks
- **Verify data flow**: Check that operations consume correct SSA values
- **CHECK-NOT**: Ensure unwanted operations/attributes are not present

## Documentation Style
- **Tone**: Formal and technical; avoid second person ("you/your")
- **Voice**: Use present tense descriptive style ("provides", "enables", "includes")
- **Structure**: Keep sentences clear and concise; end with periods
- **Content**: Explain what and why; avoid unnecessary fluff
- **Code examples**: Include complete, runnable examples where appropriate
- **References**: Follow LLVM documentation style: https://llvm.org/docs/

## Python Packaging
- **New Python packages must ship in the wheel.** When adding a new directory
  with `__init__.py` under `python/`, register it in three places, or it will
  build locally and fail in the wheel:
  1. `python/CMakeLists.txt` -- add a `declare_mlir_python_sources(...)` group
     listing every source file under the new directory.
  2. `setup.py` -- add the dotted package name to `packages=[...]` and the
     source path to `package_dir={...}`.
  3. `packaging/sim/setup.py` -- if the package is consumed by the simulator
     (`python/sim/...` or anything `ttl.sim` imports), add a `shutil.copytree`
     in `stage()` and the dotted name to `packages=[...]`.
- **Keep all runtime Python under the `ttl` namespace.** Do not introduce
  top-level packages (siblings of `ttl`) -- they pollute the public namespace
  and are easy to miss in wheel packaging. Place shared/backend-neutral code
  under `ttl._<name>` (e.g., `ttl._pipenets`).

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
- Prefer `git mv` to deleting and adding files that are in git. Stop and ask user to do if you can't do it.
- Generate commit messages and PR summaries in plain ASCII format using github markdown. When appropriate, include plain ASCII diagrams.
- Follow LLVM coding standards: https://llvm.org/docs/CodingStandards.html
- Follow best practices: https://llvm.org/docs/ProgrammersManual.html
