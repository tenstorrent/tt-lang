# tt-lang
TT-Lang is a Python-based DSL that enables authoring of programs for TT hardware.

See [RFC document](https://docs.google.com/document/d/1T8Htv9nfdufebajJidzYHfDfsSnru_9xn_whdyxn6Po/edit?usp=sharing).

## Build System

tt-lang uses a CMake-based build system that reuses tt-mlir's environment and toolchain. See [BUILD_SYSTEM.md](BUILD_SYSTEM.md) for complete documentation.

## Quick Start

### Prerequisites

tt-lang depends on tt-mlir. The required tt-mlir commit is tracked in `third-party/tt-mlir.commit`.

### Setup and Build

CMake will automatically handle the tt-mlir dependency:
- If tt-mlir is already built and found, it will use it
- If not found, it will fetch and build from the version specified in `third-party/tt-mlir.commit`

**Basic build:**

```bash
# Configure and build
cmake -GNinja -Bbuild .
cmake --build build
```

**Specify tt-mlir location explicitly:**

```bash
cmake -GNinja -Bbuild . -DTTMLIR_DIR=/path/to/tt-mlir/build/lib/cmake/ttmlir
cmake --build build
```

**With environment activation (if using pre-built tt-mlir):**

```bash
source env/activate
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

## Developer Guidelines

### Code Formatting with Pre-commit

tt-lang uses [pre-commit](https://pre-commit.com/) to automatically format code and enforce style guidelines before commits.

#### Installation

Install pre-commit using pip:

```bash
pip install pre-commit
```

Or using your system package manager:
```bash
# macOS
brew install pre-commit

# Ubuntu/Debian
sudo apt install pre-commit
```

#### Setup

After cloning the repository, install the git hook scripts:

```bash
cd /path/to/tt-lang
pre-commit install
```

This will configure git to run pre-commit checks before each commit.

#### Usage

Once installed, pre-commit will automatically run when you commit:

```bash
git commit -m "Your commit message"
```

Pre-commit will:
- Format Python code with [Black](https://github.com/psf/black)
- Format C++ code with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) (LLVM style)
- Remove trailing whitespace
- Ensure files end with a single newline
- Check YAML and TOML syntax
- Check for large files

If pre-commit makes changes, the commit will be stopped. Review the changes, stage them, and commit again:

```bash
git add -u
git commit -m "Your commit message"
```

#### Manual Formatting

To run pre-commit checks manually on all files:

```bash
pre-commit run --all-files
```

To run on specific files:

```bash
pre-commit run --files path/to/file1.py path/to/file2.cpp
```

#### Skipping Pre-commit (Not Recommended)

In rare cases where you need to skip pre-commit checks:

```bash
git commit --no-verify -m "Your commit message"
```

**Note:** CI will still run these checks, so skipping locally may cause CI failures.
