# tt-lang
TT-Lang is a Python-based DSL that enables authoring of programs for TT hardware. 

See [RFC document](https://docs.google.com/document/d/1T8Htv9nfdufebajJidzYHfDfsSnru_9xn_whdyxn6Po/edit?usp=sharing).

## Build System

tt-lang uses a CMake-based build system that reuses tt-mlir's environment and toolchain. See [BUILD_SYSTEM.md](BUILD_SYSTEM.md) for complete documentation.

## Quick Start

### Prerequisites

tt-lang depends on tt-mlir. The required tt-mlir commit is tracked in `third-party/tt-mlir.commit`.

### Setup and Build

```bash
# 1. Sync tt-mlir to the required version and build it
./scripts/sync-tt-mlir.sh

# 2. Activate environment (sources tt-mlir's environment)
source env/activate

# 3. Configure and build tt-lang
cmake -GNinja -Bbuild .
cmake --build build
```

### Updating tt-mlir Version

tt-lang tracks its tt-mlir dependency version in `third-party/tt-mlir.commit`. This file contains a git commit SHA or tag that specifies which version of tt-mlir to use.

To update to a different tt-mlir version:

```bash
# 1. Find the desired commit or tag in tt-mlir repo
cd /Users/bnorris/tt/tt-mlir  # or wherever your tt-mlir is
git log --oneline -20          # view recent commits
# or
git tag                         # view available tags

# 2. Update tt-lang's dependency file with the new SHA or tag
cd /Users/bnorris/tt/tt-lang
echo "abc123def456" > third-party/tt-mlir.commit
# or for a tag:
echo "v1.2.3" > third-party/tt-mlir.commit

# 3. Sync and rebuild tt-mlir at the new version
./scripts/sync-tt-mlir.sh

# 4. Rebuild tt-lang
source env/activate
cmake --build build
```

**Note:** The `sync-tt-mlir.sh` script will:
- Check out the specified commit/tag in your tt-mlir installation
- Warn you if there are uncommitted changes
- Rebuild tt-mlir automatically

**Current tt-mlir version:** Check `third-party/tt-mlir.commit` to see which version tt-lang is using.
