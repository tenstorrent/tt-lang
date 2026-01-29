# tt-lang Docker Container

Welcome to the tt-lang containerized environment!

## Quick Start

The tt-mlir toolchain is installed in `$TTMLIR_TOOLCHAIN_DIR` (typically `/opt/ttmlir-toolchain`).

tt-lang examples are available in `$TTMLIR_TOOLCHAIN_DIR/examples`.

Try running an example:
```bash
python $TTMLIR_TOOLCHAIN_DIR/examples/demo_one.py
```

## Available Tools

- **Editors**: `vim`, `nano`
- **Python**: `python` (aliased to `python3.11`)
- **Testing**: `pytest`

## Documentation

Full documentation: https://docs.tenstorrent.com/tt-lang

## Running Tests

```bash
cd $TTMLIR_TOOLCHAIN_DIR/test
pytest
```

## Installed Locations

- **Toolchain**: `$TTMLIR_TOOLCHAIN_DIR` (typically `/opt/ttmlir-toolchain`)
- **Examples**: `$TTMLIR_TOOLCHAIN_DIR/examples`
- **Tests**: `$TTMLIR_TOOLCHAIN_DIR/test`
- **Python packages**: `$TTMLIR_TOOLCHAIN_DIR/python_packages`
  - `ttl` - tt-lang API
  - `pykernel` - Kernel generation utilities
  - `sim` - Simulator
  - `ttmlir` - tt-mlir Python bindings
  - `ttnn` - TT-NN runtime
- **Environment**: Activated automatically via `$TTMLIR_TOOLCHAIN_DIR/env/activate`

## Mounting Your Code

To work with your own code, mount it when starting the container:

```bash
docker run -it \
  -v $(pwd):/root/my-code \
  --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-lang/tt-lang-ci-ubuntu-22-04:latest
```

Your code will be in `~/my-code`.

## Available Container Images

**Remote (from ghcr.io):**
- `ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest` - Pre-built tt-lang for users (recommended)
- `ghcr.io/tenstorrent/tt-lang/tt-lang-ci-ubuntu-22-04:latest` - tt-mlir toolchain for CI workflows
- `ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest` - Development tools for building tt-lang

**Local (if built locally):**
- `tt-lang-dist:local` - Pre-built tt-lang (recommended for users)
- `tt-lang-ci:local` - tt-mlir toolchain (for building tt-lang)
- `tt-lang-ird:local` - Development image

## Notes

- The environment is automatically activated when the container starts
- All tt-lang dependencies are pre-installed
- Examples can be copied and modified: `cp -r $TTMLIR_TOOLCHAIN_DIR/examples ~/my-examples`
