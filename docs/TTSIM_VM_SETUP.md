# VM-Based Hardware Testing with ttsim

This guide explains how to run tt-lang hardware tests on macOS using a Linux virtual machine with the ttsim simulator.

## Overview

tt-lang tests marked with `# REQUIRES: ttnn` need access to Tenstorrent hardware or a compatible simulator. Since the tt-mlir runtime only works on Linux, macOS users need a Linux VM to run these tests.

The `tools/vm/` infrastructure provides:
- **Lima** (automated) or **UTM** (manual) VM setup
- **ttsim** integration for hardware simulation
- **Shared directories** so you can edit code on macOS and test in Linux
- **Parameterized configuration** for easy customization

## Quick Start

### Prerequisites

1. **Lima** (recommended) or **UTM**:
   ```bash
   brew install lima
   ```

2. **Repository layout** - The scripts expect this structure (configurable via `TT_ROOT`):
   ```
   $TT_ROOT/                # Default: ~/tt
   ├── tt-lang/             # This repository (includes bundled ttsim binaries)
   └── tt-mlir/             # tt-mlir repository
   ```

   Note: The ttsim simulator binaries are bundled in `tools/vm/bin/` - no separate ttsim-private clone is required.

### One-Command Setup

```bash
cd $TT_ROOT/tt-lang    # or ~/tt/tt-lang with default TT_ROOT
./tools/vm/setup-vm.sh
```

This will:
1. Create a Lima VM with Ubuntu 22.04
2. Mount `$TT_ROOT` into the VM
3. Install all dependencies
4. Build tt-mlir (with runtime) and tt-lang
5. Set up the simulator environment (using bundled ttsim binaries)

Initial setup takes 30-60 minutes for the first build.

### Running Tests

```bash
# Run all tests
./tools/vm/run-tests.sh

# Run specific test target
./tools/vm/run-tests.sh check-ttlang-python-lit

# Run single test file
./tools/vm/run-tests.sh test/python/simple_add.py
```

## Configuration

### Default Configuration

The default configuration in `tools/vm/config.sh` works for the standard repository layout:

| Setting | Default | Description |
|---------|---------|-------------|
| `TT_ROOT` | `~/tt` | Root directory containing repositories |
| `VM_NAME` | `ttlang-ttsim` | VM instance name |
| `VM_CPUS` | `8` | Number of CPU cores |
| `VM_MEMORY` | `16GiB` | RAM allocation |
| `VM_DISK` | `128GiB` | Disk size |
| `UBUNTU_VERSION` | `22.04` | Ubuntu version |
| `CHIP_TYPE` | `wh` | Chip to simulate (wh=Wormhole, bh=Blackhole) |

### Custom Configuration

Create `tools/vm/config.local.sh` for personal settings:

```bash
# Copy the example
cp tools/vm/config.example.sh tools/vm/config.local.sh

# Edit with your settings
vim tools/vm/config.local.sh
```

Example customizations:

```bash
# Non-standard repository location
export TT_ROOT="/path/to/my/repos"

# More resources for faster builds
export VM_CPUS=12
export VM_MEMORY="24GiB"

# Use Blackhole chip instead of Wormhole
export CHIP_TYPE="bh"
```

### Environment Variables

All settings can also be passed as environment variables:

```bash
# Run with Blackhole chip
CHIP_TYPE=bh ./tools/vm/run-tests.sh

# Use custom VM name
VM_NAME=my-test-vm ./tools/vm/setup-vm.sh
```

## VM Management

### Status and Access

```bash
# Check VM status
./tools/vm/setup-vm.sh --status

# SSH into VM
limactl shell ttlang-ttsim

# Or with custom VM name
limactl shell $VM_NAME
```

### Inside the VM

Once in the VM, activate the ttsim environment:

```bash
source ~/activate-ttsim.sh

# Now you can run tests directly
cd $TT_ROOT/tt-lang
cmake --build build-linux --target check-ttlang-all
```

### Rebuilding After Code Changes

Code changes on macOS are immediately visible in the VM (shared directory). To rebuild:

```bash
# SSH into VM
limactl shell ttlang-ttsim

# Activate environment
source ~/activate-ttsim.sh

# Rebuild tt-lang
cd $TT_ROOT/tt-lang
cmake --build build-linux

# Run tests
cmake --build build-linux --target check-ttlang-all
```

### Deleting the VM

```bash
./tools/vm/setup-vm.sh --delete
```

## Test Categories

| Target | Description | Requires Hardware |
|--------|-------------|-------------------|
| `check-ttlang-mlir` | MLIR dialect tests | No |
| `check-ttlang-python-lit` | Python lit tests | Yes (some) |
| `check-ttlang-pytest` | pytest tests | Yes (some) |
| `check-ttlang-python-bindings` | Python binding tests | No |
| `check-ttlang-all` | All tests | Yes (some) |

Tests are marked with `# REQUIRES: ttnn` if they need hardware/simulator.

## Troubleshooting

### VM Won't Start

```bash
# Check Lima status
limactl list

# View VM logs
limactl shell ttlang-ttsim -- journalctl -xe

# Delete and recreate
./tools/vm/setup-vm.sh --delete
./tools/vm/setup-vm.sh
```

### Build Failures

**tt-mlir nanobind errors**: Random build failures are common. Retry 2-3 times:
```bash
limactl shell ttlang-ttsim
cd $TT_ROOT/tt-mlir
cmake --build build-linux
```

**Out of memory**: Increase VM memory or reduce parallelism:
```bash
# In config.local.sh
export VM_MEMORY="24GiB"

# Or limit build parallelism inside VM
cmake --build build-linux -j4
```

### Mount Issues

If changes on macOS aren't visible in VM:
```bash
# Check mount status inside VM
limactl shell ttlang-ttsim -- mount | grep tt

# Remount if needed
limactl shell ttlang-ttsim -- sudo mount -a
```

### Simulator Issues

```bash
# Verify simulator setup inside VM
limactl shell ttlang-ttsim
ls -la ~/sim/
# Should show: libttsim.so, soc_descriptor.yaml

# Check environment
source ~/activate-ttsim.sh
echo $TT_METAL_SIMULATOR
echo $TT_METAL_SLOW_DISPATCH_MODE
```

### Re-provisioning

If you need to rebuild everything:
```bash
# Remove provisioning markers
limactl shell ttlang-ttsim -- rm -rf ~/.tt-provision

# Re-run provisioning (use your TT_ROOT path)
limactl shell ttlang-ttsim -- bash \$TT_ROOT/tt-lang/tools/vm/provision-vm.sh
```

## Architecture

### Directory Layout

The VM setup uses several directories on both the macOS host and inside the Linux VM.

**On macOS (host):**

```
$TT_ROOT/                          # Default: ~/tt - shared with VM
├── tt-lang/                       # This repository
│   ├── build/                     # macOS build output
│   ├── build-linux/               # Linux VM build output (created by VM)
│   └── tools/vm/                  # VM infrastructure scripts
│       └── bin/                   # Bundled ttsim binaries
│           ├── wh/libttsim.so     # Wormhole simulator
│           └── bh/libttsim.so     # Blackhole simulator
├── tt-mlir/                       # tt-mlir repository
│   ├── build/                     # macOS build output
│   ├── build-linux/               # Linux VM build output (created by VM)
│   └── third_party/tt-metal/      # tt-metal fetched here
│       └── src/tt-metal/          # tt-metal source (after configure)
└── linux-vm/                      # Linux-specific artifacts (persistent)
    └── ttmlir-toolchain/          # LLVM/MLIR toolchain (Linux ARM64 binaries)

~/.lima/$VM_NAME/                  # Lima VM disk and metadata (macOS only)
```

**Inside the VM:**

```
$TT_ROOT/                          # Mount of host's TT_ROOT (read-write)
├── tt-lang/                       # Same as host - edits sync both ways
│   ├── build-linux/               # Linux build output
│   └── tools/vm/bin/              # Bundled ttsim binaries (from repo)
├── tt-mlir/
│   └── build-linux/               # Linux build output
└── linux-vm/                      # Linux-specific artifacts
    └── ttmlir-toolchain/          # LLVM/MLIR toolchain (built from source)

~/sim/                             # SIM_DIR - simulator runtime files
├── libttsim.so                    # Copied from bundled tools/vm/bin/
└── soc_descriptor.yaml            # Copied from tt-metal

~/.tt-provision/                   # Provisioning state markers
├── toolchain_built                # Marker: ttmlir-toolchain built
├── tt_metal_deps                  # Marker: tt-metal deps installed
├── tt_mlir_configured             # Marker: tt-mlir cmake configured
├── ttsim_patches                  # Marker: patches applied
├── tt_mlir_built                  # Marker: tt-mlir build complete
└── tt_lang_built                  # Marker: tt-lang build complete

~/activate-ttsim.sh                # Generated environment activation script
```

**Key points:**

- `$TT_ROOT` is **shared** between macOS and VM - edits on either side are immediately visible
- Linux builds use `build-linux/` directories to avoid conflicts with macOS `build/` directories
- `$TT_ROOT/linux-vm/` stores Linux-specific artifacts (toolchain) that persist across VM recreations
- The toolchain builds LLVM/MLIR from source (takes 1-2 hours on first run) and is reused on subsequent runs
- ttsim binaries are **bundled** in `tools/vm/bin/` - no separate ttsim-private clone needed
- `$SIM_DIR` (default `~/sim/`) is VM-local and contains the simulator library and SOC descriptor
- `~/.tt-provision/` tracks provisioning progress for idempotent re-runs
- Lima stores VM disk images at `~/.lima/$VM_NAME/` on macOS

### Script Files

```
tools/vm/
├── config.sh           # Central configuration (defaults)
├── config.example.sh   # Example customizations
├── config.local.sh     # Your local settings (gitignored)
├── lima.yaml.template  # Lima VM configuration template
├── lima.yaml           # Generated Lima configuration (gitignored)
├── setup-vm.sh         # VM creation/management
├── provision-vm.sh     # In-VM build script
├── run-tests.sh        # Test execution wrapper
├── utm-setup.md        # UTM manual setup guide
├── bin/                # Bundled ttsim binaries
│   ├── wh/libttsim.so  # Wormhole simulator binary
│   └── bh/libttsim.so  # Blackhole simulator binary
└── patches/            # tt-metal patches (if needed)
```

### Build Order

1. **ttmlir-toolchain** - LLVM/MLIR built from source (1-2 hours, cached for reuse)
2. **tt-mlir configure** - Downloads tt-metal via ExternalProject
3. **Apply patches** - Any ttsim-required patches to tt-metal
4. **tt-mlir build** - Builds tt-metal and tt-mlir runtime
5. **Simulator setup** - Copy bundled libttsim.so and SOC descriptor
6. **tt-lang build** - Links against runtime-enabled tt-mlir
7. **System descriptor** - Generated by ttrt for tests

### Building the Toolchain

The LLVM/MLIR toolchain is built using `tools/build-toolchain.sh`:

```bash
export TTMLIR_TOOLCHAIN_DIR=/path/to/toolchain
./tools/build-toolchain.sh /path/to/tt-mlir [build-dir]
```

The script:
- Sources tt-mlir's `env/activate` for environment setup
- Configures and builds `env/CMakeLists.txt` which fetches and builds LLVM/MLIR
- Installs to `TTMLIR_TOOLCHAIN_DIR`

Alternatively, when using CMake's FetchContent to build tt-mlir automatically, enable `TTLANG_BUILD_TTMLIR_TOOLCHAIN=ON`:

```bash
export TTMLIR_TOOLCHAIN_DIR=/path/to/toolchain
cmake -GNinja -Bbuild . -DTTLANG_BUILD_TTMLIR_TOOLCHAIN=ON
```

This calls `tools/build-toolchain.sh` automatically before building tt-mlir.

### Environment Variables

The simulator requires these environment variables:

```bash
TT_METAL_SIMULATOR=$SIM_DIR/libttsim.so   # Default: ~/sim/libttsim.so
TT_METAL_SLOW_DISPATCH_MODE=1
TTLANG_HAS_DEVICE=1
```

These are set automatically when you source `~/activate-ttsim.sh`.

## UTM Alternative

If Lima doesn't work for your setup, use UTM with manual configuration:

```bash
./tools/vm/setup-vm.sh --utm
```

This displays UTM setup instructions. For the full guide, see [tools/vm/utm-setup.md](../tools/vm/utm-setup.md).

## Known Limitations

1. **Slow dispatch only**: ttsim requires `TT_METAL_SLOW_DISPATCH_MODE=1`
2. **ARM64 only**: VMs run ARM64 Linux on Apple Silicon (no x86)
3. **Initial build time**: First tt-mlir build takes 30-60 minutes
4. **Disk space**: ~100GB required for full toolchain
5. **Mount performance**: Lima's reverse-sshfs can be slow; use `virtiofs` for better performance

## Contributing

When adding new hardware tests:
1. Mark tests requiring hardware with `# REQUIRES: ttnn`
2. Test locally with the VM setup before submitting
3. Document any new simulator requirements
