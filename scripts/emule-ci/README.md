# Compiler/emulator CI measurement

This experiment builds the pinned emulator runtime and selected compiler on one
standard GitHub-hosted runner, then measures single-device compiler execution.
It uses the tests' existing assertions, expected failures, and ordered ME2E
stages. It does not replace normal compiler or hardware CI checks.

## Scope and prerequisites

The inventory comes from the selected revision's Python pytest, ME2E, and
configured Python lit suites. Pytest excludes `multi_device` and `compile_only`.
Some remaining tests only compile or inspect host-side behavior. The pytest
observer records calls and successful returns from `ttnn.generic_op` separately
from test outcomes. A returned call does not imply that a numerical comparison
passed. Lit has separate outcomes and is not covered by that Python observer.

The LLVM/MLIR-only, Python bindings, packaging, and Python simulator suites are
outside this execution benchmark. Normal CI remains responsible for them.
Passing this benchmark does not mean that every compiler test runs on emulation.

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
remain runner-local; the worker prints its validated numeric/boolean summary.
Preflight failures also expose an allowlisted category for imports, CPU Torch,
the allocation API, dependency consistency, or timeout; exception text is not
copied from private logs.
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
in-file ordering. Python lit files are separate groups. Results include per-group
elapsed time and per-test events. Group timings include interpreter startup,
collection, fixture, compilation, execution, and reference-check costs.

## Local balancing analysis

The harness retains its existing local analysis support:

```sh
python /workspace/scripts/emule-ci/benchmark.py balance \
  --manifest /emule-results/collection/manifest.json \
  --results /emule-results/serial/results.json \
  --shards 3 --output /emule-results/plan.json \
  --public-source /workspace
```

Balancing requires complete, uncensored timings for the exact inventory. It
assigns whole files to the lightest shard, preserving original group order.
The plan contains the inventory digest, tracked group paths, and estimated times,
not runtime-generated case IDs. Inventory drift fails rather than omitting tests.

Hosted sharding is disabled. Independently rebuilt runtime images can have
different image IDs, which the current inventory digest deliberately treats as
different runtimes. A future multi-runner design needs separately reviewed
provenance and measurements; this pilot does not distribute private images or
plans to other runners. Predicted totals exclude runtime and compiler setup and
are not measured hosted performance.

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
