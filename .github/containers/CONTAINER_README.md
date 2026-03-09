# tt-lang Docker Container

Welcome to the tt-lang containerized environment! The environment is automatically activated on startup.

## Quick Start

```bash
python $TTLANG_TOOLCHAIN_DIR/examples/tutorial/multicore_grid_auto.py
```

## Installed Locations

- **Toolchain**: `$TTLANG_TOOLCHAIN_DIR` (`/opt/ttlang-toolchain`)
- **Examples**: `$TTLANG_TOOLCHAIN_DIR/examples`
- **Tests**: `$TTLANG_TOOLCHAIN_DIR/test`
- **Simulator**: `ttlang-sim` (run `ttlang-sim <script.py>` to simulate without hardware)
- **Python packages**: `$TTLANG_TOOLCHAIN_DIR/python_packages`

## Available Tools

`vim`, `nano`, `python` (3.12), `pytest`, `tt-smi`, `capture-release`, `csvexport-release` (Tracy profiler)

## Documentation

https://docs.tenstorrent.com/tt-lang
