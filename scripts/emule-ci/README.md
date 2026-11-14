# Compiler test inventory on tt-emule

`.github/workflows/tt-lang-emule-inventory.yml` builds the pinned emulator
runtime from the private emulator repository on one GitHub-hosted runner
(`ubuntu-24.04`, 4 CPUs, 16 GB), installs the compiler against it, and runs the
complete compiler test inventory serially, with tt-emule executing the
compiler-generated kernels. It does not replace normal compiler or hardware CI
and is not a required check. The tt-lang simulator ("sim") area owns failures,
the xfail list, and stack-pin bumps.

| Suite | Collection | Execution |
| --- | --- | --- |
| `mlir` | `llvm-lit --show-tests` on `test/ttlang` | one lit group, compiler-only |
| `bindings` | `llvm-lit --show-tests` on `test/bindings/python`, plus its pytest file | one lit group, plus one pytest group run last, compiler-only |
| `packaging` | `pytest --collect-only test/packaging` | one pytest group, host-only |
| `pytest` | `pytest --collect-only test/python` | one group per file, emulator |
| `me2e` | `pytest --collect-only test/me2e` | one group per file, emulator |
| `lit` | `llvm-lit --show-tests` on `test/python` | one serial lit group per file, emulator |
| `examples` | `pytest --collect-only test/examples` (the scripts and tags of `compile-and-run-examples.sh`) | one pytest group, emulator, run last |
| `tutorial` | `pytest --collect-only test/tutorial` | one pytest group, emulator, run last |

**When.** After every merge to `main`, and on manual dispatch from `main`. One
run at a time: a newer merge waits, and only the newest pending run is kept. It
never runs on pull requests, because only code already reviewed and merged to
`main` may run next to the private source and its credential. A cold build plus
the inventory is expected to take about 3.5 to 4 hours (not yet measured on
hosted runners); the job limit is 6 hours. The job summary publishes per-suite
counts only. The private checkout's Git output, build output, logs, and test IDs
stay on the runner and are deleted when the job ends; to investigate a failure,
check out that `main` commit and follow "Reproduce locally" below.

**Runner memory.** The emulator reserves device memory larger than the hosted
runner's RAM and touches only what kernels use. The kernel's default heuristic
refuses such a reservation when no swap covers it, so the job sets
`vm.overcommit_memory=1` on the runner, as Docker Desktop does; the worker's
container memory limit still bounds actual use.

**Image build downloads.** Nothing is downloaded once the private source is
present. tt-metal and its submodules (at their pinned commits), tt-metal's CPM
dependencies and SFPI toolchain, and the Python environment are downloaded in
Docker stages that never receive the private source, and the final stage
installs its apt packages before it copies anything from the build stage. Every
command after the private source arrives runs with `--network=none`. Python
wheels are pinned by URL and SHA-256 hash in
`.github/containers/emule-requirements.lock`, and the CPM dependencies and SFPI
toolchain by Git commit or content digest in
`.github/containers/emule-metal-deps.lock`; a download that differs fails the
build. `scripts/update-tt-lang-emule-locks.sh`
regenerates both lock files without the private source.

**Outcomes.** Every planned case ends in exactly one of `passed`, `failed`,
`skipped`, `xfail`, `unsupported`, `incomplete`, or `not_run`. `unsupported` is
a `multi_device` pytest skip or a lit `UNSUPPORTED` predicted at collection;
any other unexpected lit code, or a pytest case that started without a verdict
(for example after an emulator abort), is `incomplete`. `not_run` counts cases
that never started, such as after a crash or at the worker deadline. Each case
is limited to 300 s and each group to 45 minutes; a group that writes no
output or case event for 10 minutes is stopped early. A group with incomplete
coverage, or one stopped by either limit, is an `infrastructure_error`, never a
pass. Before the inventory, one representative case must pass with at least
one `ttnn.generic_op` call returning. If it does not, the summary gives one
fixed reason, such as `source_git_failed`, `case_failed`, `group_killed` or
`runtime_not_observed`, and, if the case started, its counts; other exit codes
are reported as `smoke_failure`. `test/sim` is outside this inventory.

**xfail list.** `config/tt-lang-emule-xfail.json` lists known emulator failures
by inventory case ID; only this job applies it. Each entry has a generic
`reason` (at most 160 characters of letters, digits, spaces and `,.;:()'_-`,
so no paths, URLs, or `#N` references; review and the public-content check keep
issue and repository names out) and `run`; use `run: false` for crashes, hangs,
and failures that destabilize the emulator for later cases in the same file,
which are then never executed. A new failing case in a file whose other
failures are not run is listed the same way, unless repeated runs of the whole
file show that it fails cleanly. Listed passes are counted as
`xpass_count`, and listed cases that were not collected as `stale_xfail_count`;
both are reported, not fatal, and should be pruned. Refresh the list, and the
image lock files, on every stack-pin bump.

**Admin setup.** Create the environment `tt-lang-emule`, restricted to the
`main` branch, with "Allow administrators to bypass" off. OSPO creates the
read token and stores it as the environment secret
`TTLANG_EMULE_SOURCE_READ_TOKEN`; developers do not create GitHub tokens for
workflows. Ask OSPO for repository permission Contents: Read-only (Metadata:
Read-only comes with it) on only the private emulator repository, and no other
permissions. The validation step accepts the raw value of a fine-grained token
(`github_pat_...`) or a GitHub App token (`ghs_...`), with no prefix and no
trailing newline; it rejects a classic token (`ghp_...`), which cannot be
limited to one repository. Store the repository as
`TTLANG_EMULE_SOURCE_REPOSITORY` in the form `tenstorrent/<name>` (no URL, no
trailing newline); any other value fails the validation step. Do not create
repository- or organization-level copies. The `Validate compiler-backed
emulation candidate` workflow reads the private source through its own
repository-level variable `TTLANG_EMULE_SOURCE_REPOSITORY` and secret
`TTLANG_EMULE_SOURCE_TOKEN`; this job never reads them, and this setup neither
creates nor revokes them. A repository variable is not masked in logs.

When the token expires or is revoked, every run fails at "Checkout the pinned
private source without retaining credentials" while validation still passes;
ask OSPO for a new one. To stop this job's access at once, ask OSPO to revoke
the token.

**Recommended `main` ruleset hardening** (not enforced by this workflow):
enable "Require review from Code Owners" and "Require approval of the most
recent reviewable push", and add `.github/CODEOWNERS` entries that route
`.github/containers/`, `scripts/emule-ci/`, `config/tt-lang-emule-*.json`, and
`requirements*.txt` to the sim area's owner group.

**Follow-ups.** Build parallelism; more runners if the run gets too slow.

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
  --output "$(mktemp -d)/inventory"
```

`summary.json` holds the validated counts-only summary; detailed logs, test
IDs, and manifests stay under `private/`.
