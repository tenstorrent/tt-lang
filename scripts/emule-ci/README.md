# Compiler/emulator CI measurement

The manual `tt-lang-emule-benchmark.yml` workflow builds the pinned emulator
runtime on one GitHub-hosted runner, installs the compiler against it, and runs
the complete compiler test inventory serially, with tt-emule executing the
compiler-generated kernels. It does not replace normal compiler or hardware CI.

| Suite | Collection | Execution |
| --- | --- | --- |
| `mlir` | `llvm-lit --show-tests` on `test/ttlang` | one lit group, compiler-only |
| `bindings` | `llvm-lit --show-tests` on `test/bindings/python` | one lit group, compiler-only |
| `packaging` | `pytest --collect-only test/packaging` | one pytest group, host-only |
| `pytest` | `pytest --collect-only test/python` | one group per file, emulator |
| `me2e` | `pytest --collect-only test/me2e` | one group per file, emulator |
| `lit` | `llvm-lit --show-tests` on `test/python` | one serial lit group per file, emulator |

No case is excluded by marker. Every planned case ends in exactly one outcome:
`passed`, `failed`, `skipped`, `xfail`, `unsupported`, `incomplete`, or
`not_run`. `unsupported` is a `multi_device` pytest skip or a lit `UNSUPPORTED`
that the collection-time `llvm-lit --no-execute` probe predicted; any other lit
`UNSUPPORTED`, `UNRESOLVED`, `TIMEOUT`, or unknown code is `incomplete`.
`not_run` counts planned cases without a recorded outcome, such as after a
crash or a lost report. A group whose coverage is incomplete is an
`infrastructure_error`, never a pass. The compiler-only suites prove nothing
about emulation, so the job summary reports each suite separately. `test/sim`
is outside this inventory.

## Reproduce locally

Build the runtime image from a pinned emulator checkout, then run the worker
from a clean, committed compiler checkout. The output directory must be new and
outside the checkout:

```sh
TTLANG_EMULE_BUILD_ONLY=1 TTLANG_EMULE_REBUILD=1 \
TTLANG_EMULE_RUNTIME_SOURCE_DIR=/absolute/pinned-emulator-checkout \
TTLANG_EMULE_IMAGE=ttlang-emule-ci:runner-local \
bash scripts/tt-lang-emule-container.sh
python3 scripts/emule-ci/worker.py --image ttlang-emule-ci:runner-local \
  --worker serial --output "$(mktemp -d)/serial" --timeout-seconds 13200
```

`summary.json` holds the validated counts-only summary; detailed logs, test
IDs, and manifests stay under `private/`. The owner-approval scope, required
repository configuration, and limits are described in the PR that adds this
workflow.
