# tt-lang Docker Container

Welcome to the tt-lang containerized environment!

## Quick Start

You're currently in `/root`. tt-lang examples are in `./examples/`

Try running an example:
```bash
python examples/demo_one.py
```

## Available Tools

- **Editors**: `vim`, `nano`
- **Python**: `python` (aliased to `python3.11`)
- **Testing**: `pytest`

## Documentation

Full documentation: https://docs.tenstorrent.com/tt-lang

## Running Tests

```bash
cd /opt/ttmlir-toolchain/test
pytest
```

## Installed Locations

- **Toolchain**: `/opt/ttmlir-toolchain`
- **Examples**: `/opt/ttmlir-toolchain/examples` (also copied to `~/examples`)
- **Tests**: `/opt/ttmlir-toolchain/test`
- **Python packages**: `/opt/ttmlir-toolchain/python_packages`
  - `ttl` - tt-lang API
  - `pykernel` - Kernel generation utilities
  - `sim` - Simulator
  - `ttmlir` - tt-mlir Python bindings
  - `ttnn` - TT-NN runtime
- **Environment**: Activated automatically via `/opt/ttmlir-toolchain/env/activate`

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
- `ghcr.io/tenstorrent/tt-lang/tt-lang-ci-ubuntu-22-04:latest` - Pre-built tt-lang for users
- `ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest` - Alias for ci (same image)
- `ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:latest` - Development (a few extra tools)

**Local (if built locally):**
- `tt-lang-ci:local` or `tt-lang-dist:local` - Pre-built tt-lang
- `tt-lang-ird:local` - Development image

## Notes

- The environment is automatically activated when the container starts
- All tt-lang dependencies are pre-installed
- Examples can be modified and re-run directly from `~/examples`
