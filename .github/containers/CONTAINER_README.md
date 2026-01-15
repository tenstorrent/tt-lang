# tt-lang Docker Container

Welcome to the tt-lang containerized environment!

## Quick Start

tt-lang examples are in `./examples/`

Try running an example:
```bash
python examples/demo_one.py
```

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
- **Environment**: Activated automatically via `/opt/ttmlir-toolchain/env/activate`

## With Hardware

When running with Tenstorrent hardware, the container must be started with device access:

```bash
docker run -it \
  --device=/dev/tenstorrent/0 \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  tt-lang-ci:latest
```
