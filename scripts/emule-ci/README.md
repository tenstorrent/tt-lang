# Compiler/emulator CI measurement

This experiment builds the pinned emulator runtime and selected compiler on one
standard GitHub-hosted runner, then runs the complete compiler test inventory
with tt-emule executing the compiler-generated kernels. It uses the tests'
existing assertions, expected failures, and ordered ME2E stages. It does not
replace normal compiler or hardware CI checks.

## Scope and prerequisites

The inventory is the same six suites `check-ttlang-all` runs, collected from
the selected revision with no marker selection:

| Suite | Collection | Execution |
| --- | --- | --- |
| `mlir` | `llvm-lit --show-tests` on `test/ttlang` | one lit group, compiler-only, parallel |
| `bindings` | `llvm-lit --show-tests` on `test/bindings/python` | one lit group, compiler-only, parallel |
| `packaging` | `pytest --collect-only test/packaging` | one pytest group, host-only, invoked like `check-ttlang-packaging` (rootdir `test/`, so IDs read `packaging/…`) |
| `pytest` | `pytest --collect-only test/python` | one pytest group per file, emulator execution |
| `me2e` | `pytest --collect-only test/me2e` | one pytest group per file, ordered stages, emulator execution |
| `lit` | `llvm-lit --show-tests` on `test/python` | one serial lit group per file, emulator execution |

Cases are never excluded by marker. `compile_only` cases run normally; they do
not launch kernels. `multi_device` cases run and skip themselves on the
single-device P150 profile. Every planned case ends in exactly one outcome:
`passed`, `failed`, `skipped`, `xfail`, `unsupported`, `incomplete`, or
`not_run`. `unsupported` means any skip of a `multi_device` pytest case (the
skip reason itself stays private) or a lit `UNSUPPORTED` verdict for a test
that lit already reported unsupported during collection (`llvm-lit
--no-execute`, which evaluates `REQUIRES`/`UNSUPPORTED` directives without
running anything). A lit `UNSUPPORTED` anywhere else, an `UNRESOLVED`, a
`TIMEOUT`, or an unrecognized lit code is `incomplete` and makes the group an
`infrastructure_error`, so a runtime whose `ttnn`/`tt-device` features went
missing cannot turn the Python lit suite green. Collection also refuses to
build an inventory unless `llvm-lit --show-suites` lists both features.
`not_run` counts planned cases without a recorded outcome: after a native
abort, a killed group, or a lost or unreadable report, the whole group is
still accounted for. A group whose coverage is incomplete is an
`infrastructure_error`, never a pass.

The pytest observer records calls and successful returns from `ttnn.generic_op`
separately from test outcomes, and records each collected case's `multi_device`
and `compile_only` markers. It imports the runtime only for the device suites
(`TTLANG_EMULE_CI_OBSERVE_RUNTIME=1`); collection and the packaging suite never
import it. A returned call does not imply that a numerical comparison passed.
Lit has separate outcomes and is not covered by that observer. The
compiler-only suites prove nothing about emulation; the job summary reports
them per suite so that emulator coverage is not inferred from their passes.

The Python simulator suite (`test/sim`) tests `tt-lang-sim` itself and is outside
this inventory; `call-test-sim.yml` remains responsible for it.

The runtime must match the selected compiler's LLVM and supported stack pins,
contain CPU-only Torch and test requirements, and support the compiler's required
TTNN allocation APIs. Before compiler installation, a bounded preflight checks
imports, CPU-only Torch, `pip check`, and the range-lockstep allocation binding.
It does not open a device. The worker then builds the compiler and
`ttlang-test-tools` and runs `test/python/pipe/test_broadcast_2d.py`. This
documented multicast smoke test must pass its numerical assertion and return from
`ttnn.generic_op`. Its cache is separate from the subsequent full inventory.
Compatibility failure stops the job; it does not skip unsupported APIs or change
dependency pins.

## Runner-local approval and manual pilot

The emulator owner approved runner-local builds and possible log exposure.
This scope does not authorize publication of private source, images, binaries,
artifacts, or shared caches. The manual workflow builds the private runtime on
the same runner that builds and tests the compiler. It reuses the pinned public
toolchain base image; it does not rebuild LLVM from scratch. No private runtime
image is pulled from or pushed to a registry.

After the native build, the packaging stage creates a fresh CPU-only Python
environment using the existing toolchain preparation and CPU guard helpers.
Subsequent requirements installation and final interpreter relocation repeat
the guard. The public donor's Torch variant is not assumed to be CPU-only.
Native dependency pins remain unchanged.

The workflow has one serial execution job and a no-permission authorization job.
The authorization job requires the expected repository and experiment branch,
an acknowledgement of the recorded owner decision, and a reviewed full source
SHA. It accepts no arbitrary PR or fork ref. Actions are pinned by commit.
The following configuration is required before activation:

| Configuration | Purpose |
| --- | --- |
| `TTLANG_EMULE_APPROVED_SHA` repository variable | Full reviewed compiler/workflow commit, matching the selected event SHA. |
| `TTLANG_EMULE_SOURCE_REPOSITORY` repository variable | Approved private source repository, in `tenstorrent/NAME` form. |
| `TTLANG_EMULE_SOURCE_READ_TOKEN` Actions secret | Credential restricted to reading the private source repository contents. |

The default `GITHUB_TOKEN` only grants access to the current repository; it is
not a replacement for private cross-repository access. The source credential is
passed only to the pinned private checkout action. Both checkouts disable
credential persistence, and a verification step rejects retained local Git
authentication configuration before any build. The private checkout is outside
the compiler tree and is never mounted into test containers. The existing
launcher validates both pins and archives the exact emulator commit into the
Docker build context, without its Git metadata.

The reviewed-SHA variable is an operational interlock, not an additional
per-commit owner approval policy or a sandbox against repository writers who
can change the workflow. Normal repository review applies. This code does not
provision secrets or variables, grant access, merge changes, or dispatch jobs.
GitHub's [manual-dispatch rules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)
require the workflow file to be present on the default branch before activation.

The initial pilot uses a fresh ephemeral Ubuntu 24.04 standard runner. Its
runtime build has a 120-minute step limit (115-minute command timeout plus
cleanup allowance) and a 2 GiB initial free-space check. Build progress, elapsed
time, exit status, and final disk usage appear in the job log. This initial
check does not guarantee that the native build fits the runner's disk or memory.
The compiler/test worker has a 220-minute deadline, a 2 GiB sampled disk-free
floor, four CPU and 12 GiB memory limits, no network, and fresh owned build/JIT
volumes. The job's total limit is six hours. Timeouts, resource exhaustion,
collection errors, test failures, and incomplete coverage remain failures.

Only job logs leave the runner. Build logs can contain runtime diagnostics under
the recorded approval. Detailed test logs, generated code, test IDs, and manifests
remain runner-local; the worker prints its validated numeric/boolean summary, and
a final step renders that summary's per-suite table (planned, run and complete
groups, planned cases, case outcomes and group statuses per suite) into the job
summary, even when an earlier step failed. Preflight failures also expose an
allowlisted category for imports, CPU Torch, the allocation API, dependency
consistency, or timeout; exception text is not copied from private logs.

The 220-minute worker deadline was sized for a measurement pilot. The complete
inventory took 2 h 5 min on an 8-CPU/32 GiB CPU-only Linux container on
October 2, 2026; its cost on a four-CPU standard runner is what the first run
measures. Groups run compiler-only suites first, then the device suites, and
results are recorded per group, so an exhausted deadline still reports which
suites completed. A deadline exhaustion is a failed run, not partial credit.
There are no artifact uploads, shared Actions/BuildKit caches, image publication,
or build-record exports. Private runner-local evidence is lost when the runner
is discarded. Automatic PR triggers, additional workers, paid runners, and
automatic retries are not part of this pilot.

## Run one bounded local worker

The existing image must have verified source/dependency provenance and exist in
the local Docker daemon. The worker resolves its immutable image ID and uses the
existing installer and launcher. It does not build or download an image. The
experiment source must be locally committed and clean before measurement.

```sh
runtime_image=VERIFIED_EXISTING_LOCAL_RUNTIME
experiment_dir="$(mktemp -d "${TMPDIR:-/tmp}/ttlang-emule-ci.XXXXXX")"
python3 scripts/emule-ci/worker.py \
  --image "$runtime_image" --output "$experiment_dir/serial" \
  --worker serial --cpus 4 --memory-gib 12
```

The output is a new directory outside the checkout. `--dry-run` prints the stage
plan without invoking Docker. Local defaults are four CPUs, 12 GiB RAM, a
45-minute limit, and a 10 GiB host disk-free floor. Sampling approximates peaks;
on macOS the disk-free floor describes the output filesystem, not the Docker
VM's separate disk. Runtime build resource use is not measured by the worker.

A runtime-only build uses the existing manifest/Dockerfile owner and stops before
compiler installation:

```sh
TTLANG_EMULE_BUILD_ONLY=1 TTLANG_EMULE_REBUILD=1 \
TTLANG_EMULE_RUNTIME_SOURCE_DIR=/absolute/pinned-emulator-checkout \
TTLANG_EMULE_IMAGE=ttlang-emule-ci:runner-local \
bash scripts/tt-lang-emule-container.sh
```

`summary.json` contains bounded measurements. Verbose evidence stays under
`private/`; its `identity.json` records the newly created, labeled build/cache
volumes, which remain available for inspection. Only task-owned processes and
containers may be stopped during local measurement.

## Collect and measure

After activating the installed compiler/emulator environment, collection and
execution use the same source revision and immutable runtime identity:

```sh
revision="$(git -C /workspace rev-parse HEAD)"
runtime_id=sha256:REPLACE_WITH_VERIFIED_IMAGE_ID
python /workspace/scripts/emule-ci/benchmark.py collect \
  --source /workspace --build /ttlang-build \
  --revision "$revision" --runtime-id "$runtime_id" \
  --output /emule-results/collection
python /workspace/scripts/emule-ci/benchmark.py run \
  --source /workspace --build /ttlang-build \
  --revision "$revision" --runtime-id "$runtime_id" \
  --manifest /emule-results/collection/manifest.json \
  --output /emule-results/serial
```

Output locations are new directories. Existing evidence is preserved. Collection
errors, timeouts, native crashes, missing results, and incomplete coverage remain
errors. Failed tests are not retried or converted to successful timing results.

Whole pytest files stay together, preserving ordered ME2E prerequisites and normal
in-file ordering. Python lit files are separate groups. The MLIR, Python-binding
and packaging suites are one group each, run before the device suites. Results
include per-group elapsed time, per-suite membership, normalized case outcomes
and per-test events. Group timings include interpreter startup, collection,
fixture, compilation, execution, and reference-check costs. The manifest is
version 2 (`selection: complete`); version 1 manifests with marker selection are
rejected rather than silently re-run with narrower coverage. All groups run
serially on one worker; hosted sharding is not implemented.

## Evidence and interpretation

Record the compiler revision, local image identity, dependency pins, requirement
hashes, CPU/RAM limits, cold build/setup times, test times, cache sizes, and sampled
worker resource peaks. Sampling can miss brief peaks. Local Docker measurements
do not establish a standard hosted runner's disk fit or speed.

Host-side scripts, dependencies, and the runtime are trusted inputs. GitHub's
[secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
also cautions that output redaction is not a complete disclosure boundary. The
controls are not a sandbox for malicious source code. The absence of a hosted
run or a full test result must not be reported as a pass.

Compatibility failures remain explicit. The benchmark does not promote
dependency pins, bypass APIs, or relax numerical tolerances to obtain a green run.
