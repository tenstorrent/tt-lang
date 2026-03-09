# tt-lang IRD (Development Container)

This container has the tt-mlir toolchain pre-built at `$TTLANG_TOOLCHAIN_DIR`.
tt-lang is **not** pre-built -- clone and build it yourself.

## Quick Start

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_USE_TOOLCHAIN=ON
source build/env/activate
cmake --build build
```

To test your build, you can run all tests with `ninja -C build check-ttlang-all`.

## Running Tests

```bash
source build/env/activate
ninja -C build check-ttlang-mlir     # MLIR lit tests
ninja -C build check-ttlang-pytest   # Python pytest tests
```

## Toolchain Contents

- **Toolchain root**: `$TTLANG_TOOLCHAIN_DIR` (`/opt/ttlang-toolchain`)
- **MLIR/LLVM tools**: `$TTLANG_TOOLCHAIN_DIR/bin/` (ttmlir-opt, mlir-opt, etc.)
- **Python venv**: `$TTLANG_TOOLCHAIN_DIR/venv/` (Python 3.12, dev dependencies pre-installed)

## Available Tools

`vim`, `nano`, `tmux`, `ssh`, `sudo`, `git`, `ccache`, `tt-smi`, `capture-release`, `csvexport-release` (Tracy profiler)

## Documentation

Full documentation: https://docs.tenstorrent.com/tt-lang
