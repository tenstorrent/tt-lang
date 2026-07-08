# Third-party dependencies

tt-lang's external dependencies are managed as git submodules.

## Submodules

| Submodule | Description |
|-----------|-------------|
| `llvm-project` | LLVM/MLIR (pinned to a specific commit) |
| `tt-metal` | TT-Metal runtime (pinned to a specific tag) |

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

Submodule uplift instructions are maintained in the build docs:
[Uplifting Submodules](../docs/sphinx/build.md#uplifting-submodules).

That section covers updating LLVM and tt-metal, rebuilding the toolchain,
validating the uplift, and committing the resulting pointer changes.
