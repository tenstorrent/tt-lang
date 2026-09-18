# Compiler-backed simulator validation: September 17, 2026

This directory preserves the completed six-suite validation and subsequent
failure investigation. The run began on September 17 UTC and ended on
September 18 UTC. Results: **5,765 passed, 165 failed, 65 skipped/unsupported,
and 42 expected failures** across 6,037 cases, with no test errors.

## Reports

- [Results, root-cause triage, and recommended next steps](TRIAGE.md)
- [Every failing test, itemized](failure-itemization.md)
- [Machine-readable failure inventory](failure-itemization.csv)
- [Full sweep summary](summary.json) and [sanitized host invocation](invocation.json)
- [Compiler/emulator interface investigation](ROOT_CAUSE_NOTES.md)
- [Multicast source-level investigation](LOOPBACK_SOURCE_NOTES.md)
- [Sparse buffer metadata and mixed-operation numerical evidence](NUMERICAL_STATIC_CONFIG_NOTES.md)
- [Tree-reduction and multicast numerical evidence](NUMERICAL_TREE_LOOPBACK_NOTES.md)
- [Two-process numerical probe summary](numerical-probe/summary.json),
  [first run](numerical-probe/run-001/summary.json), and
  [second run](numerical-probe/run-002/summary.json)

The per-case subdirectories under the two probe runs also preserve ten captured
`compiled-metadata.json` files. These record the buffer allocation and kernel
core ranges used to investigate the numerical failures.

## Exact scope

The full sweep and both probes tested clean compiler commit
`70c5d093df06d211aa78332447bbd8d602d2ba79` from branch
`kostas/tt-lang-sim-simple-cli`, with emulator
`7292395ce55a208a8ede0a4635a9f2167c8c4939` and tt-metal
`b6c508c4790fac0a11597e43a5421624cb461553`.

Execution used Docker Desktop on a Mac, a `linux/amd64` runtime, and one emulated
Blackhole P150. It did not use the QB or physical TT hardware. The local runtime
image tag `tt-lang-emule:compiler-suite-7292395` is not a public download URL.
The immutable image ID and complete runtime provenance are in the reports.

The later documentation/help-test commit `9f3d2a079c5ced5dda2e4a7dd11b96f7137de357`
and this report publication do not represent additional compiler-suite runs.
The suite includes compile-only and rejection tests; its pass count is not a
count of emulator executions. It does not qualify full models or multi-device
execution. Proposed compiler/emulator fixes have not been implemented or
validated with a patched runtime.

## Raw evidence archive

Large raw logs, JUnit XML, tensors, generated kernels, and diagnostic helper
scripts are intentionally not committed. A local archive outside the worktree
preserves them, including the superseded interim notes:

```text
tt-lang-simulator-validation-archive/20260917T233554Z-pyfnmtqs.tar.gz
SHA-256: 1b64a489e5858d315a0bcd748c017600224662c71ae3f5377e65954daff4ca9d
```

This is an archive filename, not a downloadable repository artifact. The
archive is retained by the report author; reproducing tensor postprocessing
or inspecting cited raw log lines requires obtaining that archive. It contains
`reports/20260917T233554Z-pyfnmtqs/` and `diagnostics/`, relative to a checkout's
`.ttlang-sim/` directory. Extract into an empty inspection directory first;
restoring into a checkout must not overwrite other reports.

Raw filenames and log-line references in the inventory refer to that archive.
Container paths in captured metadata describe the tested runtime, not files
provided by this compact Git export. The local diagnostic wrapper in the
triage reproduction example also requires the archived `diagnostics/` directory.

## Export policy

This is a curated, sanitized export, not a byte-for-byte copy of every raw
artifact. Personal host/interpreter paths were replaced with placeholders,
private dependency utility references were generalized, and relative links and
historical wording were updated for publication. Recorded revisions, test
identities, outcomes, numerical measurements, runtime paths, and captured
compiled metadata were preserved. The raw archive retains the original paths.

The original 165-row inventory was independently reconciled against all JUnit
failures. Evidence distinguishes specific diagnostics from inferred family-level
causes; the 89 generic JIT failures must not be treated as individually proven
instances of a more specific error.

For installation and ordinary test commands, start with the
[new-user guide](../../../sphinx/simulator-getting-started.md).
