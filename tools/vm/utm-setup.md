# UTM Manual Setup Guide

This guide provides step-by-step instructions for setting up a UTM virtual machine on macOS for running tt-lang hardware tests with the ttsim simulator.

Use this guide if you prefer UTM over Lima, or if Lima is not working for your setup.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- At least 32GB RAM recommended (16GB minimum)
- 150GB+ free disk space
- Administrative access for installing packages in the VM

## Step 1: Install UTM

Download and install UTM from: https://mac.getutm.app/

UTM is free and provides native virtualization on Apple Silicon using the Apple Virtualization framework.

## Step 2: Download Ubuntu Image

Download the Ubuntu 22.04 ARM64 server image:

- Direct link: https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04.5-live-server-arm64.iso

Save the ISO to a known location (e.g., `~/Downloads/`).

## Step 3: Create the Virtual Machine

1. Launch UTM and click the **+** button to create a new VM

2. Select **Virtualize** (not Emulate - emulation is unusably slow)

3. Choose **Linux** from the preconfigured options

4. Configure the following settings:
   - **Use Apple Virtualization**: Check this box (required for good performance)
   - **Boot ISO Image**: Select the Ubuntu ISO you downloaded
   - **Enable Rosetta**: Optional, allows running x86_64 binaries

5. **Hardware Configuration**:
   - **Memory**: 16384 MB (16GB) minimum, 24576 MB (24GB) recommended
   - **CPU Cores**: Use default (all available) or at least 8 cores

6. **Storage**:
   - **Size**: 128 GB minimum (tt-mlir + tt-metal need ~80GB)

7. **Shared Directory** (important for development):
   - Add a shared directory pointing to your TT repositories
   - **Directory**: Select your `~/tt` folder (or wherever your repos are)
   - **Read Only**: Uncheck (must be writable for builds)

8. **Name**: Give your VM a descriptive name (e.g., `ttlang-ttsim`)

9. Click **Save** to create the VM

## Step 4: Install Ubuntu

1. Start the VM by clicking the play button

2. Complete the Ubuntu installation:
   - Select your language and keyboard layout
   - Choose **Ubuntu Server** installation
   - Configure network (use defaults)
   - Configure storage (use entire disk)
   - Set up your user account
   - **Important**: Enable **OpenSSH server** during installation
   - Skip additional snaps

3. After installation completes, reboot the VM

4. Remove the installation ISO:
   - Stop the VM
   - In UTM, edit the VM settings
   - Remove the CD/DVD drive or clear the ISO
   - Start the VM again

## Step 5: Initial VM Configuration

SSH into your VM or use the UTM console:

```bash
# Get the VM's IP address
ip a
# Look for the inet address under enp0s1 (e.g., 192.168.64.x)

# From your Mac, SSH in:
ssh <username>@<vm-ip>
```

### Install Essential Packages

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    libhwloc-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    software-properties-common
```

### Install Clang 17+

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo add-apt-repository -y "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-17 main"
sudo apt update
sudo apt install -y clang-17 lld-17

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
```

### Increase Swap Space

```bash
sudo swapoff /swap.img || true
sudo rm -f /swap.img
sudo fallocate -l 16G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
echo '/swap.img none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 6: Mount Shared Directory

UTM shared directories use VirtioFS. Mount your shared directory:

```bash
# Create mount point
mkdir -p ~/tt

# Mount the shared directory (replace 'share' with your share name)
sudo mount -t virtiofs share ~/tt

# Make it permanent
echo 'share /home/$(whoami)/tt virtiofs rw,nofail 0 0' | sudo tee -a /etc/fstab
```

**Note**: The share name is typically `share` unless you named it differently in UTM settings.

Verify the mount:
```bash
ls ~/tt
# Should show: tt-lang, tt-mlir, ttsim-private
```

## Step 7: Run Provisioning

Navigate to the tt-lang tools directory and run the provisioning script:

```bash
cd ~/tt/tt-lang/tools/vm

# Set environment variables
export TT_ROOT_VM="$HOME/tt"
export CHIP_TYPE="wh"  # or "bh" for Blackhole
export SIM_DIR="$HOME/sim"

# Run provisioning
./provision-vm.sh
```

This will:
1. Build ttsim
2. Configure and build tt-mlir with runtime support
3. Apply any necessary patches
4. Set up the simulator directory
5. Build tt-lang
6. Create an activation script at `~/activate-ttsim.sh`

**Note**: The initial build takes 30-60 minutes depending on your hardware.

## Step 8: Verify Setup

After provisioning completes:

```bash
# Activate the ttsim environment
source ~/activate-ttsim.sh

# Verify environment
echo $TT_METAL_SIMULATOR
# Should output: /home/<user>/sim/libttsim.so

# Run a simple test
cd ~/tt/tt-lang
cmake --build build --target check-ttlang-mlir
```

## Step 9: Run Hardware Tests

```bash
source ~/activate-ttsim.sh
cd ~/tt/tt-lang

# Run all tests
cmake --build build --target check-ttlang-all

# Run Python lit tests only
cmake --build build --target check-ttlang-python-lit

# Run a specific test
llvm-lit -sv test/python/simple_add.py
```

## Troubleshooting

### Shared Directory Not Mounting

If VirtioFS mount fails:
1. Ensure the shared directory is configured in UTM VM settings
2. Check that the share name matches (default is `share`)
3. Try mounting with verbose output: `sudo mount -t virtiofs -v share ~/tt`

### Build Failures

**nanobind errors**: tt-mlir builds can fail randomly due to nanobind. Retry 2-3 times:
```bash
cd ~/tt/tt-mlir
cmake --build build
```

**Out of memory**: Increase swap or reduce parallel jobs:
```bash
cmake --build build -j4  # Limit to 4 parallel jobs
```

### SSH Connection

To simplify SSH access, add to your Mac's `~/.ssh/config`:
```
Host ttlang-vm
    HostName <vm-ip>
    User <username>
    StrictHostKeyChecking no
```

Then connect with: `ssh ttlang-vm`

### VM Performance

For best performance:
- Use Apple Virtualization (not QEMU backend)
- Allocate at least 8 CPU cores
- Use 24GB RAM if available
- Keep the VM running rather than stopping/starting frequently

## Running Tests from macOS

Unlike Lima, UTM doesn't have a native CLI integration. To run tests from macOS:

```bash
# SSH into VM and run tests
ssh ttlang-vm "source ~/activate-ttsim.sh && cd ~/tt/tt-lang && cmake --build build --target check-ttlang-all"
```

Or create a simple wrapper script on your Mac:
```bash
#!/bin/bash
ssh ttlang-vm "source ~/activate-ttsim.sh && cd ~/tt/tt-lang && $*"
```

## Useful Commands

```bash
# Check VM status from UTM GUI or:
# (UTM doesn't have CLI, check the app)

# Rebuild after code changes
source ~/activate-ttsim.sh
cd ~/tt/tt-lang
cmake --build build

# Rebuild tt-mlir if needed
cd ~/tt/tt-mlir
source env/activate
cmake --build build

# View simulator logs (if issues)
export TT_METAL_LOGGER_LEVEL=DEBUG
# Then run tests
```

## Next Steps

- Read [docs/VM_TESTING.md](../../docs/VM_TESTING.md) for detailed testing documentation
- Check the [tt-lang TESTING.md](../../test/TESTING.md) for test writing guidelines
- Review ttsim README for simulator-specific information
