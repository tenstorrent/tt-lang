# TT-Lang CI Workflows

This document describes the CI/CD workflows for TT-Lang.

## Quick Reference

### Running CI on a PR

CI runs automatically on pull requests. The `on-pr.yml` workflow:
1. Calls toolchain workflow first (builds cache if needed, skips if exists)
2. Waits for toolchain to complete
3. Runs the build workflow

### Rebuilding the Toolchain Cache

The toolchain (LLVM + tt-mlir) is cached to speed up CI. Rebuild it when:
- Updating `third-party/tt-mlir.commit` to a new version
- Cache was evicted (not accessed for 7+ days)
- Cache appears corrupted

**To rebuild:**
1. Go to **Actions** > **Build tt-mlir Toolchain (Cache)**
2. Click **Run workflow**
3. Optionally check "Force rebuild even if cache exists"
4. Wait for completion (~3-4 hours)

The cache is also automatically rebuilt:
- Nightly at 6 AM UTC (keeps cache warm, rebuilds if tt-mlir.commit changed)

### Building Docker Images

Docker images require the toolchain cache to exist.

**To build:**
1. Ensure toolchain cache exists for the tt-mlir commit
2. Go to **Actions** > **Build Docker Image**
3. Click **Run workflow**

### Cache Key Format

```
Linux-ttlang-toolchain-v1-{tt-mlir-commit-sha}
```

The cache key is based on the tt-mlir commit SHA from `third-party/tt-mlir.commit`. The `v1` suffix distinguishes this cache from legacy cache formats.

## Troubleshooting

### CI fails with "cache miss"

This shouldn't happen normally since `on-pr.yml` calls the toolchain workflow first. If it does:

**Solution:**
1. Re-run the failed workflow (toolchain should build automatically)
2. Or manually trigger: **Actions** > **Build tt-mlir Toolchain (Cache)**

### Container build fails with "cache miss"

The toolchain cache doesn't exist. Run the toolchain workflow first, or trigger a PR CI run which will build the cache.

### Cache was evicted

GitHub evicts caches not accessed in 7 days. The nightly schedule prevents this.

**Solution:** Run the toolchain workflow manually.

---

## Architecture Details

<details>
<summary>Click to expand workflow architecture</summary>

### Overview

TT-Lang CI uses a **dedicated cache workflow** pattern ([GitHub Actions best practice](https://github.com/actions/cache#skipping-steps-based-on-cache-hit)):
1. A dedicated workflow builds and caches the expensive toolchain (LLVM + tt-mlir)
2. Other workflows restore from this cache using `fail-on-cache-miss: true`

```
+-------------------------------+
|         on-pr.yml             |  <-- PR orchestrator
+-------------------------------+
              |
              | calls (workflow_call)
              v
+-------------------------------+
|  call-build-ttmlir-toolchain  |  <-- Checks cache, builds if needed
|  (check-cache on ubuntu-latest|      (large runner only if cache miss)
|   build on large runner)      |
+-------------------------------+
              |
              | saves to cache (if built)
              v
+-------------------------------+
|  GitHub Actions Cache         |
|  Linux-ttlang-toolchain-v1-{sha} |
+-------------------------------+
              |
              | waits (needs: toolchain)
              v
+-------------------------------+
|  call-build.yml               |  <-- Builds tt-lang using cached toolchain
|  (uses build-ttlang action)   |
+-------------------------------+
              |
              v
+-------------------------------+
|  Container Build (separate)   |
|  (Docker images)              |
+-------------------------------+
```

### Workflow Files

| File | Purpose |
|------|---------|
| `on-pr.yml` | PR orchestrator - calls toolchain, then build |
| `on-push.yml` | Push trigger |
| `call-build-ttmlir-toolchain.yml` | Cache builder - LLVM + tt-mlir (skips if cache exists) |
| `call-build.yml` | CI build and test |
| `call-build-docker.yml` | Container image builds |
| `call-test-hardware.yml` | Hardware tests |

### `call-build-ttmlir-toolchain.yml` (Cache Builder)

**Purpose:** Builds and caches the LLVM + tt-mlir toolchain.

**Triggers:**
- Called by `on-pr.yml` via `workflow_call` (ensures cache exists before build)
- Manual dispatch (for force rebuild or testing specific commits)
- Nightly schedule (keeps cache warm)

**Jobs:**
1. **`check-cache`** (ubuntu-latest, ~1 min): Checks if cache exists, outputs `needs-build`
2. **`build-toolchain`** (large runner, ~4-6 hours): Only runs if `needs-build == true`

**What it does:**
1. Determines tt-mlir commit from `third-party/tt-mlir.commit`
2. Checks if cache exists (on standard runner)
3. If cache miss: builds LLVM toolchain + tt-mlir (on large runner)
4. Saves to cache

### `call-build.yml` (CI)

**Purpose:** Builds and tests TT-Lang.

**Triggers:**
- Pull requests
- Push to main
- Manual dispatch
- Daily schedule

**What it does:**
1. Restores toolchain from cache (`fail-on-cache-miss: true`)
2. Builds TT-Lang using `build-ttlang` reusable action
3. Runs tests (smoketest, MLIR lit, Python bindings, Python lit)
4. Archives artifacts for hardware tests

**Timeout:** 60 minutes

### `call-build-docker.yml` (Container Build)

**Purpose:** Builds Docker images.

**Triggers:**
- Version tags (`v*.*.*`)
- Manual dispatch
- Called by other workflows

**What it does:**
1. Checks if Docker images already exist
2. Restores toolchain from cache (`fail-on-cache-miss: true`)
3. Validates cache contents
4. Builds tt-lang **outside Docker** (for space efficiency)
5. Creates `ttlang-install` directory (toolchain + tt-lang)
6. Builds Docker images using two build contexts:
   - `ttmlir-toolchain` - for dev image
   - `ttlang-install` - for user image
7. Pushes to GitHub Container Registry

**Timeout:** 120 minutes

### Cache Contents

```
ttmlir-toolchain/
+-- bin/                    # FileCheck, llvm-lit, ttmlir-*, etc.
+-- lib/
|   +-- cmake/
|   |   +-- ttmlir/         # TTMLIRConfig.cmake
|   |   +-- mlir/           # MLIRConfig.cmake
|   |   +-- llvm/           # LLVMConfig.cmake
|   +-- *.a, *.so           # Libraries
+-- include/                # Headers
+-- venv/                   # Python virtual environment
+-- python_packages/
|   +-- ttmlir/             # tt-mlir Python bindings
|   +-- ttrt/runtime/ttnn/  # TTNN runtime
+-- tt-metal/               # tt-metal runtime
```

### Cache Lifecycle

| Event | Action |
|-------|--------|
| PR opened/updated | `on-pr.yml` calls toolchain workflow, builds cache if needed |
| Nightly schedule (6 AM UTC) | Toolchain workflow runs, keeps cache warm |
| Cache not accessed for 7 days | GitHub evicts cache (prevented by nightly schedule) |
| Manual dispatch with `force_rebuild=true` | Cache rebuilt even if exists |

### Build Times

| Workflow | Cache Hit | Cache Miss |
|----------|-----------|------------|
| Toolchain (LLVM + tt-mlir) | Skip (uses cache) | ~3-4 hours |
| CI (TT-Lang only) | ~15-30 minutes | N/A (fails) |
| Container (TT-Lang in Docker) | ~10-15 minutes | N/A (fails) |

### Reusable Actions

| Action | Purpose |
|--------|---------|
| `.github/actions/build-ttlang` | Validates toolchain, configures, and builds tt-lang |

### Scripts

| Script | Purpose |
|--------|---------|
| `.github/scripts/check-toolchain-cache.sh` | Checks cache and prints helpful error on miss |
| `.github/scripts/determine-ttmlir-commit.sh` | Reads tt-mlir commit |
| `.github/scripts/normalize-ttmlir-install.sh` | Replaces symlinks with files |
| `.github/containers/cleanup-toolchain.sh` | Removes unnecessary binaries |
| `.github/containers/build-docker-images.sh` | Docker build orchestration |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `TTMLIR_TOOLCHAIN_DIR` | Path to toolchain |
| `TT_LANG_HOME` | TT-Lang repository root |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Parallel build jobs (default: 2) |

</details>

## Related Documentation

- [Container README](containers/README.md) - Docker image documentation
- [Build System](../docs/BUILD_SYSTEM.md) - Build system architecture
- [Testing](../test/TESTING.md) - Test documentation
- [actions/cache best practices](https://github.com/actions/cache) - GitHub Actions caching
