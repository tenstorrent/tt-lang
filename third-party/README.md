# Third-party dependencies

tt-lang's external dependencies are managed as git submodules.

## Submodules

| Submodule | Description |
|-----------|-------------|
| `tt-mlir` | TT-MLIR compiler infrastructure (pinned to specific commit) |
| `llvm-project` | LLVM/MLIR (pinned to version matching tt-mlir) |
| `tt-metal` | TT-Metal runtime (pinned to version matching tt-mlir) |

## Switching branches

After checking out a branch that pins different submodule commits:

```bash
git submodule update --init --force --recursive --depth 1
```

`--force` is required because CMake applies patches (from `patches/`) to the
submodule working trees at configure time. Without it, git refuses to overwrite
the patched files. The patches are re-applied automatically on the next
configure.

Then reconfigure and rebuild:

```bash
cmake -G Ninja -B build
cmake --build build
```

## Updating dependencies

To bump tt-mlir and its matched dependencies:

```bash
cd third-party/tt-mlir
git fetch origin
git checkout <new-commit>
cd ../..

# Update llvm-project to match tt-mlir/env/CMakeLists.txt LLVM_PROJECT_VERSION
cd third-party/llvm-project
git fetch origin
git checkout <llvm-commit-from-tt-mlir>
cd ../..

# Update tt-metal to match tt-mlir/third_party/CMakeLists.txt TT_METAL_VERSION
cd third-party/tt-metal
git fetch origin
git checkout <tt-metal-commit-from-tt-mlir>
cd ../..

git add third-party/tt-mlir third-party/llvm-project third-party/tt-metal
git commit -m "Bump third-party dependencies"
```
