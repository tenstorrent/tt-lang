# TT-Lang CI Workflows

This document describes the CI/CD workflows for TT-Lang.

## Quick Reference

### Running CI on a PR

CI runs automatically on pull requests. If CI fails with a cache miss error, the toolchain cache needs to be built first (see below).

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
- When `third-party/tt-mlir.commit` changes (path trigger)
- Weekly on Sunday 6 AM UTC (keeps cache warm)

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

The toolchain cache doesn't exist for this tt-mlir commit.

**Solution:**
1. Go to **Actions** > **Build tt-mlir Toolchain (Cache)**
2. Run the workflow
3. Wait for completion, then re-run CI

### Container build fails with "cache miss"

Same as above - run the toolchain workflow first.

### Cache was evicted

GitHub evicts caches not accessed in 7 days. The weekly schedule should prevent this.

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
|  call-build-ttmlir-toolchain  |  <-- Dedicated cache builder
|  (builds LLVM + tt-mlir)      |
+-------------------------------+
              |
              | saves to cache
              v
+-------------------------------+
|  GitHub Actions Cache         |
|  Linux-ttlang-toolchain-v1-{sha} |
+-------------------------------+
              |
              | restores cache
              v
+-------------------------------+
|  build-ttlang action          |  <-- Reusable composite action
|  (validates, configures,      |
|   builds tt-lang)             |
+-------------------------------+
        |             |
        v             v
+---------------+  +------------------+
|  CI Workflow  |  |  Container Build |
|  (tests)      |  |  (Docker images) |
+---------------+  +------------------+
```

### Workflow Files

| File | Purpose |
|------|---------|
| `call-build-ttmlir-toolchain.yml` | Dedicated cache builder - LLVM + tt-mlir |
| `call-build.yml` | CI build and test |
| `call-build-docker.yml` | Container image builds |
| `call-test-hardware.yml` | Hardware tests |
| `on-pr.yml` | PR trigger |
| `on-push.yml` | Push trigger |

### `call-build-ttmlir-toolchain.yml` (Cache Builder)

**Purpose:** Builds and caches the LLVM + tt-mlir toolchain.

**Triggers:**
- Changes to `third-party/tt-mlir.commit`
- Manual dispatch
- Weekly schedule (keeps cache warm)

**What it does:**
1. Checks if cache already exists (skips build if so)
2. Installs build dependencies
3. Clones tt-mlir
4. Builds LLVM toolchain (`cmake -B env/build env`)
5. Builds tt-mlir directly (`cmake --build build`)
6. Normalizes and cleans up toolchain
7. Saves to cache

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
| `third-party/tt-mlir.commit` changes | Toolchain workflow triggered, new cache created |
| Weekly schedule (Sunday 6 AM UTC) | Toolchain workflow runs, keeps cache warm |
| Cache not accessed for 7 days | GitHub evicts cache (prevented by weekly schedule) |
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
