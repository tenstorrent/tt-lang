# ttl-specialize-cores: status of the previously-failing tests

Per-core specialization is exposed as the compiler option `specialize_cores`
and is **disabled by default** (enable with `--ttl-specialize-cores`).

## Background

An earlier, single-pass version of `ttl-specialize-cores` ran at the TTL level
and cloned every kernel once per launch coordinate, const-folding the
coordinate. Enabling it for a full `test/python` run produced **41 failures**:

1. **PipeNet collectives (38 tests)** -- `RuntimeError` at compile time: `pipe
   transfer ... requires queue depth greater than 1`. Per-core cloning turned
   mutually-exclusive pipe posts into concurrent posts on one logical pipe.
2. **L1 accumulation (3 tests)** -- `AssertionError` (PCC mismatch): cloned
   L1-accumulation kernels produced silently wrong numerics.

## Resolution: two-phase, TTKernel-level specialization

The pass was redesigned into two phases (see `Passes.td`):

- **Phase A `ttl-specialize-plan`** (parallel `func.func` pass, TTL level):
  runs `LaunchNodeDomainAnalysis` and records a `ttl.specialize_plan` only for
  kernels that actually **branch** on `ttl.core_x` / `ttl.core_y`. It never
  clones. It also refuses to plan any module that uses pipes.
- **Phase B `ttl-specialize-cores`** (`ModuleOp` pass, TTKernel level, right
  before EmitC): materializes clones from the plan and forces each marked
  branch to its group's outcome; `canonicalize` / `cse` then prune dead paths.

This resolves both failure classes:

- **Pipe modules are never specialized** (Phase A's module-level pipe gate), so
  they compile and run exactly as on the default path. Phase B also runs after
  PipeNet lowering, so pipe queue-depth validation is untouched.
- **Kernels that use the coordinate only as data are not cloned** -- the
  runtime `MyLogicalX/Y` reads give each core its own coordinate, so
  L1-accumulation and addressing kernels stay single whole-grid binaries and
  keep correct numerics.

Only kernels that branch on a core coordinate (and are not in a pipe module) are
cloned; that path is covered by
`test/ttlang/Dialect/TTL/Transforms/specialize_cores.mlir`.

## How to verify

```bash
# All previously-failing tests should now pass with specialization ON:
TTLANG_COMPILER_OPTIONS="--ttl-specialize-cores" python -m pytest \
  test/python/pipe test/python/test_elementwise_l1_acc.py \
  test/python/test_matmul_l1_acc.py -q

# And the dedicated e2e test (specialization is a safe no-op for its
# data-addressing kernels):
python -m pytest test/python/test_specialize_cores.py -q
```
