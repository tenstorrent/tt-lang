# Simulator compiler-suite results and root-cause triage

The complete six-suite run finished with **5,765 passed, 165 failed,
65 skipped/unsupported, and 42 expected failures** out of 6,037 cases.
There were no test errors. All 165 failures are individually listed in the
[failure inventory](failure-itemization.md) and [CSV](failure-itemization.csv).
The CSV was independently reconciled with every JUnit report: no missing,
duplicate, or extra failing identities.

The consistent CLI and first-run documentation are implemented. The compiler
and emulator defects below are diagnosed, not fixed by this change.

## Tested state

| Component | Exact selection |
|---|---|
| Branch | `kostas/tt-lang-sim-simple-cli` |
| Compiler checkout | `70c5d093df06d211aa78332447bbd8d602d2ba79`, clean throughout the full sweep and numerical probes |
| Local Docker image | `tt-lang-emule:compiler-suite-7292395` |
| amd64 image ID | `sha256:d703049cde4b1ed3d9bd32adc9ffc27777194b747a95129c53f63bc7e9682a78` |
| Emulator | `7292395ce55a208a8ede0a4635a9f2167c8c4939` |
| Metal runtime | `b6c508c4790fac0a11597e43a5421624cb461553` |
| Platform | Mac host, Docker Desktop, `linux/amd64` runtime |
| Target | One emulated Blackhole P150 |
| Runtime manifest's compiler baseline | `59c53e209a6b871ce90937dff0cf26d8bc3e25af`, not the currently tested checkout |

This was not run on the QB or physical TT hardware. The image name is a local
tag, not a claim that it is publicly downloadable. The stack's actual external
Metal runtime was inspected; the compiler's CMake banner also prints its
default Metal pin, which is not the external runtime selected for this run.

Evidence: [host invocation](invocation.json), [complete summary](summary.json).
The container sweep ran from 2026-09-17 23:36:08 UTC to 2026-09-18 01:54:08 UTC,
approximately 2 hours 18 minutes. Its exit status was correctly nonzero.

## Results by suite

| Suite | Passed | Failed | Skipped/unsupported | Expected failures | Duration |
|---|---:|---:|---:|---:|---:|
| MLIR | 517 | 0 | 0 | 0 | 23 s |
| Python bindings | 3 | 0 | 0 | 0 | 6 s |
| Packaging | 306 | 0 | 0 | 0 | 38 s |
| Python pytest | 3,948 | 161 | 64 | 8 | 6,019 s |
| Middle-end end-to-end | 859 | 0 | 0 | 34 | 1,395 s |
| Python lit | 132 | 4 | 1 | 0 | 798 s |
| **Total** | **5,765** | **165** | **65** | **42** | |

JUnit and `summary.json` encode expected failures as skipped; the table above
separates them. Pytest's ordinary skips comprise 63 multi-device/topology
requirements and one DSL tracing limitation.

MLIR, bindings, and packaging do not execute emulator kernels. The other suites
mix compilation, rejection checks, and device execution. These totals are not
counts of emulator kernel launches. The six-suite command excludes `test/sim`
and tutorials and does not qualify full Kimi K3, dspark, all models, multi-device
execution, performance, or real hardware.

## Failure families and confidence

| Family | Cases | Diagnosis | Evidence level |
|---|---:|---|---|
| Buffer reset, reconfiguration, and reuse | 132 | Hardware RISC-V custom-LLK code reaches the x86 emulator JIT without an emulation implementation. | Specific assembly diagnostics in 56 cases; another 76 show only generic JIT failure. Family-level attribution for those 76 remains an inference. |
| Fused RMSNorm | 24 | Missing custom LLK interfaces, beginning with `ckernel_sfpu_rsqrt.h`. | Specific missing-header diagnostics in 11 cases; another 13 show only generic JIT failure. Further unsupported instruction interfaces identified statically. |
| Mixed exp/broadcast, DRAM and L1 | 2 | Shared-kernel descriptors omit a buffer absent from the first core; FP32 input then falls through to BF16 decoding. | Source discrepancy, captured allocation metadata, and exact predicted tensor fingerprint in both repeated runs. No patched-runtime A/B validation yet. |
| Generic FP32 tree reduction, DRAM and L1 | 2 | The same first-core descriptor problem affects a sparse FP32 accumulator buffer. | Captured sparse allocation metadata and host-side wrong-format model reproduce all 448 output values bit-for-bit in both repeats and memory placements; BF16/specialized controls pass. No patched-runtime A/B validation yet. |
| Cross-buffer multicast loopback | 1 | Emulator multicast semaphore increment includes the sender, which the compiler also increments locally; completion can precede the second payload. | Source-confirmed hardware/emulator semantic mismatch and exact predicted corrupt tile in both repeated runs. No patched-runtime A/B validation yet. |
| Device printing | 4 | Emulator device-print macros are no-ops, so FileCheck cannot find required output. | Direct missing-output diagnostics and pinned source implementation. |

All 156 JIT failures are individually observed as compilation failures. Only
67 have case-specific detailed Clang diagnostics in the captured logs; the
other **89 retain an explicitly generic classification** in the inventory.
Test family names alone are not proof of a detailed cause. Matching diagnostics
uses the exact parametrized failure heading plus matching traceback file and
function, not a nearby error from another test.

### 1. Buffer reset/reconfiguration: support gap

The compiler embeds the custom LLK bodies from
`include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h` and
`experimental_dfb_reconfiguration.h` verbatim in generated C++. Their RISC-V
`lw`, `sw`, and `and x0` instructions cannot compile for the emulator's x86
host. Its patcher handles standalone `fence`, not these blocks.

Removing assembly or making barriers no-ops is unsafe. Emulation combines
hardware UNPACK/PACK participants, requires cooperative waits, and must preserve
completion ordering, shared stream counters, CB geometry, high-index masks, and
unselected live buffers. Existing emulator helpers are useful but have different
barrier semantics.

Recommended boundary: TT-Lang supplies a narrowly selected emulation hook while
retaining the hardware implementation; tt-emule implements
`complete_dfb_interface_work`, `reset_dfb_interfaces`, and
`reconfigure_dfb_interfaces`. Regress repeated resets, IDs 32-63, live aliases,
capacity changes, storage switches, discard, and cached execution. Removing the
first blocker is not a promise that all 132 cases will then pass.

### 2. Fused RMSNorm: support gap

`experimental_row_normalization.h` requires `ckernel_sfpu_rsqrt.h`, which the
pinned emulator does not provide. A similarly named compatibility forwarder
does not implement the entire contract. Static inspection also finds missing
custom instruction-template interfaces; these have not been separately claimed
as reproduced compiler diagnostics.

A whole-operation `row_normalization_block` adapter is narrower than emulating
the complete custom instruction engine. Preserve optional/broadcast gamma,
tile geometry, precision boundaries, and caller-owned DST results. The existing
emulator RMSNorm routine is not a direct substitute: it requires gamma and
writes CB memory itself. Ordinary rsqrt cases pass, so this is not a blanket
absence of rsqrt support.

### 3. Shared-kernel buffer metadata: likely common cause of four wrong-result cases

The pinned Metal emulation runner derives a shared compute kernel's CB formats
and geometry from its first core only. Hardware gathers matching CBs across all
kernel core ranges. An otherwise valid sparse buffer absent from the first core
therefore gets format 255 (`Invalid`) in emulation. The pinned emulator accepts
that enum and falls through to BF16 pack/unpack behavior.

For the mixed exp/broadcast program, captured metadata shows exp input CB0 on
core `(0,0)`, output CB1 on both cores, and broadcast input **CB2** only on core
`(1,0)`. All four captures (two runs, DRAM/L1) have zero mismatches in the exp
half and exactly 992 in the broadcast half. Every broadcast row exactly equals
the independently predicted FP32-as-BF16 decoding pattern. This is not normal
floating-point approximation error.

For generic tree reduction, captured metadata shows FP32 CB0 only on cores
`(1,0)` through `(7,0)`, while the generic compute kernel covers `(0,0)` through
`(7,0)`. Generic FP32 fails in both memory placements; specialized FP32 and
generic BF16 controls pass. This makes first-core-only descriptors a concrete
emulator/runtime integration defect, not evidence that FP32 is generally
unsupported.

A host-side model of the observed tree and the erroneous BF16 accumulator
packing reproduces all 448 output float32 bit patterns exactly, with maximum
error zero, in both runs and both memory placements. This independently
corroborates the numerical mechanism without changing emulator behavior.

Recommended fix: build descriptors from all participating cores, checking for
incompatible formats/geometries rather than silently choosing one; validate
`Invalid` formats rather than silently decoding them as BF16. Add sparse-buffer
placement regressions in both memory types and generic/specialized modes.

Detailed captures and reasoning: [static-configuration notes](NUMERICAL_STATIC_CONFIG_NOTES.md)
and [tree/loopback notes](NUMERICAL_TREE_LOOPBACK_NOTES.md).

### 4. Multicast completion: correctness bug

The compiler correctly signals remote destinations with
`noc_semaphore_inc_multicast(..., num_dests=3)`, then separately signals the
sender for a four-core loopback transfer. Hardware multicast atomics exclude
the sender. The emulator helper ignores `num_dests` and includes every core in
the rectangle, including the sender. The sender's completion counter advances
twice for each payload, allowing the second receive to be published too early.

Both focused runs have exactly the predicted wrong tile: rows `32:64`, columns
`0:32`, all 1,024 values zero. The other 7,168 values equal 7; no -42 output
sentinels remain. The full sweep also had 1,024 mismatches, with maximum absolute
error 6 at `(32,0)`, consistent with a different unwritten/stale-buffer pattern;
its complete tensor was not captured. Payload loopback itself is implemented;
the mismatch is completion signaling.

Recommended fix: match hardware source-exclusion semantics for multicast
atomics, retain explicit local completion, and test repeated stripes with
different source/destination CBs and multiple scheduler configurations.
See [exact source evidence](LOOPBACK_SOURCE_NOTES.md).

### 5. Device printing: observability gap

The four exact cases are `python/dprint/cb_metadata.py`, `runtime.py`,
`tensor_accessor_page_print.py`, and `tile_after_store.py`. Their FileCheck
assertions require output discarded by the emulator's print macros.

Implement the required behavior or explicitly mark the feature unsupported.
Skipping a print test is not a passing computation test. These failures alone
do not establish numerical corruption.

## Focused reruns

Two independent pytest processes each ran the five numerical failures and five
controls: **5 failed and 5 passed in each run**, with no skips. Both processes
used the unchanged compiler checkout and pinned runtime, not patched emulator
code. Tensor capture and cached metadata inspection occur after the original
test assertion fails; allocation and execution were not monkeypatched.

Controls: gather/broadcast iteration; specialized FP32 tree reduction in DRAM
and L1; generic BF16 tree reduction in DRAM and L1.

Evidence: [probe summary](numerical-probe/summary.json),
[first run](numerical-probe/run-001/summary.json),
[second run](numerical-probe/run-002/summary.json). The tracked per-case directories preserve compiled metadata. The raw archive
also preserves CPU tensors, original outcomes, captured generated C++ text,
and tracebacks. The probe exit code correctly remains nonzero.

## Reproduction

After the existing runtime has been configured, from the repository root:

```sh
./bin/tt-lang-sim --backend=emule --test
```

For a smaller selection:

```sh
./bin/tt-lang-sim --backend=emule --test --suite pytest
```

The following diagnostic wrapper is in the local raw archive, not a fresh Git
checkout. After restoring its `diagnostics/` directory under `.ttlang-sim/`,
select a new output directory:

```sh
./bin/tt-lang-sim --backend=emule .ttlang-sim/diagnostics/numerical_probe.py -- \
  --reports-dir /workspace/.ttlang-sim/reports/numerical-rerun-001 \
  --include-controls --repeat 2
```

Run one emulator command at a time per checkout; build and JIT caches are shared.
The [new-user guide](../../../sphinx/simulator-getting-started.md) describes
Docker prerequisites and setup from an accessible image or pinned emulator
source. Diagnostic helpers and raw logs/tensors remain local ignored artifacts,
not dependencies of the simulator or committed test changes. Compact reports
and compiled metadata are versioned here; see the [archive policy](README.md).

## Recommended follow-up order

1. Fix multicast-atomic source exclusion and sparse shared-kernel descriptors,
   with focused semantic regressions. These are wrong-result defects.
2. Add the narrow TT-Lang/emulator custom-LLK boundary and real DFB
   reset/reconfiguration implementations. Do not replace synchronization with
   no-ops.
3. Add the fused row-normalization adapter and its precision/geometry tests.
4. Implement or explicitly classify device-print support.
5. Pin the repaired compatible runtime only after focused validation, then
   rerun the entire suite and retain the before/after inventory.

The historical aggregate pass count has no rigorous matched per-test baseline
available here. Differences from that count must not be called fixes or
regressions without matching revisions and reports. These diagnoses also do
not imply that repairing the first observed blocker removes all later issues.

Additional source-level detail: [interface investigation](ROOT_CAUSE_NOTES.md)
and the [complete failure inventory](failure-itemization.md).

## CLI and documentation validation

Implementation commit: `70c5d093df06d211aa78332447bbd8d602d2ba79`.
The CLI uses `--backend=emule` consistently for program execution and
`--setup`, `--smoke-test`, `--examples`, and `--test`. The old `emule` subcommand
remains a compatibility alias. The Python backend retains its lightweight path.

- Canonical Docker `--backend=emule --smoke-test`: passed, including the external
  C++ call through compiler and emulator.
- Focused CLI and compiler-suite-runner tests: 104 passed.
- Launcher/container BATS tests: 51 passed.
- Host `test/sim`, after updating one stale help-text assertion: 1,168 passed,
  58 skipped, 2 expected failures.
- Shell syntax and `git diff --check`: passed.
- Repository-wide pre-commit checks: passed.
- Reporting helpers: 18 synthetic self-tests passed; these are not device tests.

Validation gaps: the host packaging run had 305 passes and one failure because
the host environment lacks `pytest-rerunfailures`; the same packaging suite
passed all 306 cases inside the configured Docker runtime. The host lacks
`sphinx_reredirects` and `sphinx_rtd_theme`, so HTML documentation was built with
available extensions and Sphinx's classic theme instead. Existing specification
and theme-option warnings remain; shared host environments were not modified.

After all runtime diagnostics finished, only documentation refinements and the
stale simulator help-test assertion were changed. The full compiler sweep was
not repeated for those documentation/test-only changes. During that diagnostic
pass, no branch was pushed and no issue was posted. This subsequent report
publication does not claim any compiler/emulator numerical fix.

Those final refinements are committed as `9f3d2a07` on
`kostas/tt-lang-sim-simple-cli`; the working tree was clean at that point. The original
implementation and all recorded emulator executions remain tied to `70c5d093`.
