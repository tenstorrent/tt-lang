# K3 host-mock CI

The `K3 host mock` workflow dispatches private program-construction tests for an
exact public compiler commit. It runs after pushes to `main`, once daily, or by
manual dispatch from `main`. A manually supplied SHA must be a full lowercase
commit SHA reachable from public `main`. The default is the public workflow's
exact SHA. This test establishes host program construction only. Device
correctness and timing remain untested.

## Using this during TT-Lang development

Use these mocks to detect compiler/API, DFB planning, and kernel-generation
failures encountered while constructing K3 programs. Both workflows must be
merged and the credentials and execution runner configured before hosted runs
are available; see the repository settings below.

| Development task | How to run the check |
| --- | --- |
| Check a change merged into TT-Lang | A push to `main` automatically checks that exact compiler SHA; a daily run checks the workflow's `main` revision. |
| Reproduce a failure or check an earlier merged commit | Manually dispatch the public workflow from `main` with a full SHA reachable from `main`. |
| Check an unmerged TT-Lang change | Use the private local runner with a clean compiler checkout at the candidate commit; the hosted workflow rejects unmerged commits. |

### Dispatch and inspect a merged compiler commit

An authenticated repository member can use local `gh`. Replace the example SHA
with the full compiler commit to check:

```bash
compiler_sha=bc36476bdb0c348f4cb272bc7e51292c1b844444
gh workflow run k3-host-mock.yml --repo tenstorrent/tt-lang --ref main \
  -f compiler_sha="$compiler_sha"
gh api "repos/tenstorrent/tt-lang/commits/${compiler_sha}/status" \
  --jq '.statuses[] | select(.context == "K3 host mock") | {state, description, target_url}'
```

The status appears on the requested compiler commit. Its `target_url` opens the
public orchestration run. A successful result means all five pinned fixtures
reached their first mock dispatch with zero exits and valid generated-kernel
archives. This is a host-construction result; device correctness and timing
still require separate tests. The status is not a required PR merge check.

For a failure, open the public run first to distinguish dispatch/configuration
failures from private build/mock failures. With private-repository access, find
the private run matching the compiler SHA and correlation ID in its title.
Its `k3-host-mock-private-results` artifact contains provenance, per-entry
reports and logs, and generated-kernel archives, retained for seven days.

### Check a compiler candidate before merging

Private-repository members can follow the
[local reproduction instructions](https://github.com/tenstorrent/tt-lang-ops-and-models/blob/main/ci/k3_host_mock/README.md#local-reproduction)
with a clean checkout at their committed compiler candidate and the documented
pinned model/dependency checkouts. This builds that candidate and runs the same
five fixtures without hardware. Local preparation accepts a full candidate SHA
without the hosted public-main restriction. Inspect its `summary.json`, reports,
and generated kernels before continuing with device validation.

The public workflow checks out its own trusted `main` revision, creates a
repository-scoped GitHub App token, and dispatches
`tenstorrent/tt-lang-ops-and-models`'s `k3-host-mock.yml` on private `main`. It
passes the exact compiler SHA and a unique correlation ID. It polls the private
run for at most 50 minutes, requiring an exact match of compiler SHA and request
ID in the run title. The App token expires after one hour. The private workflow
builds the compiler from the requested source revision rather than using the
model's compiler submodule pin.

The public `K3 host mock` commit status is attached to the requested compiler
SHA, including when that SHA differs from the public workflow revision. Its
target URL is the public orchestration run. Public output contains only compiler
SHA, correlation ID, success/failure, and the scope limitation. Private run URLs,
reports, source code, build logs, and generated kernels remain private. Token
creation failure, dispatch failure, private failure or cancellation, and polling
timeout produce a failing status. Workflow cancellation invokes terminal status
publication when GitHub can schedule the final step; forced runner termination
can leave a pending status.

## Credentials and repository settings

An organization administrator installs a dedicated dispatch App with
`Actions: write` and the mandatory `Metadata: read` permission on the private
ops-and-models repository. `Actions: write` also permits reading workflow runs
for polling. A repository administrator creates environment
`k3-host-mock-dispatch`, selects deployment branches and tags, and permits only
the branch `main` with no tag or PR merge-ref rules. That environment contains:

- Environment variable `K3_DISPATCH_APP_ID`.
- Environment secret `K3_DISPATCH_APP_PRIVATE_KEY`.

The App key must not be a repository or organization secret: existing public CI
passes those secrets to reusable workflows with `secrets: inherit`. Environment
secrets are available only to jobs referencing the environment after its branch
restriction passes. The branch restriction is configured before adding the key;
the "protected branches only" setting is insufficient when repositories use
rulesets without classic branch protection.

Repository policy must permit this workflow's `GITHUB_TOKEN` to create commit
statuses and permit the pinned official checkout and App-token actions.

The dispatch token is requested for only the private ops-and-models repository
and only the Actions permission. It is present only in the trusted public
orchestration job, which has no pull-request trigger. It is never passed into
compiler/model execution or private artifacts. The public workflow does not use
`pull_request_target`.

The private repository additionally needs a separate App with `Contents: read`
on its private dependency repository; only its source-preparation environment
receives that credential.
Private execution also requires a configured Linux x64 Docker runner with at
least 16 GB RAM. Configuration is documented with the private runner. Both workflows must be
reviewed and merged to their default branches before dispatching by filename.
No repository secret contains a user's personal `gh` login.

The initial status is separate from `check-all-green`. Public `main` currently
requires `check-all-green` through ruleset 9293173. A new requirement for
`K3 host mock` would prevent PR merges because this initial workflow deliberately
runs only trusted commits already reachable from `main`. Any future PR-required
deployment needs a separately reviewed mechanism that preserves private-source
and credential isolation. Initial rollout does not change that ruleset.

Actions policy and App installation require administrator verification. The
implementation does not grant permissions or modify branch protection.

## Validation

The dispatch tests run in the existing script-test workflow and locally:

```bash
python3 .github/scripts/tests/test_dispatch_k3_host_mock.py
actionlint .github/workflows/k3-host-mock.yml
```

GitHub API references:

- [Reusable workflow access](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations).
- [Workflow dispatch](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event).
- [Commit status publication](https://docs.github.com/en/rest/commits/statuses#create-a-commit-status).
- [Environment secrets and branch restrictions](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
