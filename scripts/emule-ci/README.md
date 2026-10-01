# Compiler/emulator CI measurement

This experiment measures the compiler execution suites on a prepared tt-emule
runtime, then partitions the measured work into three independent workers.
It uses the compiler tests' existing assertions, expected failures, and ordered
ME2E stages. It does not replace the normal compiler or hardware CI checks.

## Scope and prerequisites

The inventory comes from the selected revision's Python pytest, ME2E, and
configured Python lit suites. Pytest excludes `multi_device` and `compile_only`.
Some remaining tests only compile or inspect host-side behavior; those markers
alone do not prove emulator execution. The pytest observer records calls and
successful returns from `ttnn.generic_op` separately from test outcomes. A
returned call does not imply that a test's numerical comparison passed. Lit
execution has its own outcomes and is not covered by that Python observer.

The LLVM/MLIR-only, Python bindings, packaging, and Python simulator suites are
outside this execution benchmark. Normal CI remains responsible for them.

The runtime image must match the selected compiler's LLVM and supported stack
pins, contain CPU-only Torch and the test requirements, and pass representative
fresh-cache emulator execution before a full benchmark. The compiler is built
from the selected source; prepared native dependencies are reused. Each worker
uses separate compiler and JIT caches and does not access physical devices.
After installation, a separately timed preparation stage builds the existing
`ttlang-test-tools` target, including `ttlang-op-stats` required by Python lit.
Collection and shard execution both require this stage to succeed.

Hosted execution requires approval for the runtime image, workflow context,
registry access, logs, retention, and budget. Image contents and verbose logs
may have different visibility requirements from this public repository. The
experiment does not grant that approval or publish an image.

The draft workflow is manual-only, restricted to the experiment branch, and
requires an affirmative company-approval input. This input records a decision;
it is not a substitute for actual authorization. Registry access and workflow
registration must be verified before dispatch. GitHub may require the new
manual workflow to exist on the default branch before it becomes dispatchable;
this experiment does not merge itself to satisfy that requirement.

The serial job produces a balanced plan only from complete timings. Its test
failure status remains visible even when a complete failing baseline supports
the three-worker comparison. Artifacts contain bounded summary/plan data with
one-day retention; verbose logs and caches are not uploaded. The pilot is
limited by operator approval to four workflow runs, rather than imposing a new
repository-wide CI or billing policy.

## Run one bounded local worker

Set `runtime_image` to an approved, validated image already present in the local
Docker daemon. The worker resolves its immutable image ID and uses the existing
installer and launcher; it does not build or download an image. The experiment
source must be locally committed and clean before measurement.

```sh
runtime_image=APPROVED_EXISTING_LOCAL_RUNTIME
experiment_dir="$(mktemp -d "${TMPDIR:-/tmp}/ttlang-emule-ci.XXXXXX")"
python3 scripts/emule-ci/worker.py \
  --image "$runtime_image" --output "$experiment_dir/serial" \
  --worker serial --cpus 4 --memory-gib 12
```

The output is a new directory outside the checkout. `--dry-run` prints the stage
plan without invoking Docker. The stages are install, prepare test tools,
collect, and test; supplied shard manifests omit collection but retain test-tool
preparation. Separate timings identify compiler setup versus test execution.

Local defaults are four CPUs, 12 GiB RAM, a 45-minute limit, and a 10 GiB host
disk-free floor. `--cpus 2 --memory-gib 4` provides a smaller local probe. The
manual hosted workflow explicitly uses a 2 GiB disk-free floor and longer
serial/shard limits. Sampling approximates peaks; on macOS the disk-free floor
describes the host output filesystem, not the Docker VM's separate disk.

`summary.json` contains bounded measurements. Verbose evidence stays under
`private/`; its `identity.json` records the newly created, labeled build/cache
volumes, which remain available for inspection. Each shard uses fresh volumes.
The three-worker design does not itself grant hosted-runner capacity or image
publication permission; availability and company approval remain prerequisites.

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

Output locations are new directories. Existing evidence is preserved.
Collection errors, timeouts, native crashes, missing results, and incomplete
coverage remain errors. No failed-test retries are introduced. A failed test
is not converted to a successful timing result.

Whole pytest files stay together, preserving ordered ME2E prerequisites and
normal in-file ordering. Python lit files are separate groups. Results include
per-group elapsed time and per-test events. Group timings include interpreter
startup, collection, fixture, compilation, execution, and reference-check costs.

## Balance three workers

```sh
python /workspace/scripts/emule-ci/benchmark.py balance \
  --manifest /emule-results/collection/manifest.json \
  --results /emule-results/serial/results.json \
  --shards 3 --output /emule-results/plan.json
```

Balancing requires a complete, uncensored baseline for the exact manifest.
It sorts groups by measured duration and assigns each to the currently lightest
shard, then restores the original group order within each shard. Validation
checks that each group appears exactly once across the shards.

Each worker passes the same manifest and plan to `run`, with `--plan` and a
different `--shard` value from 1 to 3. Each worker starts with fresh caches.
Predicted totals exclude pull/setup time and are estimates: cache reuse differs
between the serial run and independent shards. The longest indivisible file
also limits balance. Actual shard runs establish elapsed time and resource use.

## Evidence and interpretation

Record the compiler revision, image identity, dependency pins, requirement
hashes, CPU/RAM limits, cold pull and compiler setup times, test times, cache
sizes, and sampled resource peaks. Sampling can miss brief peaks; local Docker
measurements do not establish a standard GitHub runner's disk fit or speed.

Full command logs, tracebacks, JUnit output, and generated source remain private
unless approved for publication. Only explicitly selected summary fields and
public test IDs are candidates for upload. A shared Actions cache is unnecessary
for the initial cold-worker comparison.

Compatibility failures in the pinned compiler/runtime combination remain in
the results. They must be investigated independently of CPU-only packaging or
image slimming; the benchmark does not alter dependency pins, bypass APIs, or
relax numerical tolerances to obtain a green run.
