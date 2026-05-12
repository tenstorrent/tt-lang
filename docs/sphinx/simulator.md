# Functional Simulator

TT-Lang includes a functional simulator that runs operations as pure Python, without requiring Tenstorrent hardware or the full compiler stack. Use it to validate kernel logic and iterate quickly during development.

The simulator typically supports more language features than the compiler at any given point — see the [functionality matrix](specs/TTLangSpecification.md#appendix-d-functionality-matrix) for current coverage.

## Setup

The recommended path is to install the simulator from PyPI:

```bash
python3 -m venv --prompt ttlang ttlang-venv
source ttlang-venv/bin/activate
pip install tt-lang-sim
tt-lang-setup
```

See [Getting Started — Install from PyPI](getting-started.md#install-from-pypi)
for details. `tt-lang-sim` runs on Linux and macOS and does not require
Tenstorrent hardware.

To run the simulator from a source checkout instead (without building the
compiler), configure with `-DTTLANG_SIM_ONLY=ON` to create just the Python
environment:

```bash
cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON
cmake --build build
source build/env/activate
```

This skips the LLVM, tt-mlir, and tt-metal builds entirely and only sets up the Python venv with runtime dependencies.

If you have already built the full TT-Lang compiler (`source build/env/activate`), the simulator works without any additional setup.

## Running

```bash
ttlang-sim examples/eltwise_add.py
```

Run the simulator test suite:

```bash
python -m pytest test/sim/
```

Some tests are marked `slow` and skipped by default.  Pass `--run-slow` to
include them (the hardware CI always does; the GitHub-hosted sim CI does not):

```bash
python -m pytest test/sim/ --run-slow
```

## Debugging

The simulator runs as standard Python code, so any Python debugger works with it.

### VSCode

Create a debug configuration in `.vscode/launch.json`:

```json
{
  "name": "Debug TT-Lang Simulator",
  "type": "debugpy",
  "request": "launch",
  "module": "ttl.sim.ttlang_sim",
  "args": ["${file}"],
  "console": "integratedTerminal",
  "justMyCode": false,
  "cwd": "${workspaceFolder}"
}
```

1. Open a TT-NN program file in VSCode (e.g., `examples/eltwise_add.py`)
2. Set breakpoints in your program code
3. Press F5 or select "Debug TT-Lang Simulator" from the Run menu
4. The debugger stops at breakpoints, allowing variable inspection and step-through execution
