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
Tenstorrent hardware. That install adds **`ttlang-sim`** and the trace post-processor
**`ttlang-sim-stats`** to your `PATH`. There is no separate PyPI package for
statistics; `ttlang-sim-stats` ships only as a console entry point with the
simulator distributions (`tt-lang-sim`, or full `tt-lang`, which includes the same
simulator).

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

## Simulator statistics (`ttlang-sim-stats`)

Tensor, pipe, and dataflow-buffer statistics are **not** printed by `ttlang-sim`
itself. Record a JSON Lines trace with **`ttlang-sim`** using **`--trace`**
(after the script path), then pass that
file to **`ttlang-sim-stats`** to print the same summary tables (for sharing,
diffing, or inspecting a run without re-executing the kernel). The
**`ttlang-sim-stats`** command is installed together with **`tt-lang-sim`** (or
with full **`tt-lang`**); it is not distributed or installed on its own.

From a repository checkout, run **`./bin/ttlang-sim-stats`** (repo root). After
`pip install tt-lang-sim` (or `pip install tt-lang`), or `source build/env/activate`
from a **CMake** build, **`ttlang-sim-stats`** is on your **`PATH`**. The
underlying entry point is **`python -m sim_stats`**; override the interpreter
with **`PYTHON`** if needed (for example
`PYTHON=python3.12 ./bin/ttlang-sim-stats trace.jsonl`).

1. **Record a JSON Lines trace** while simulating (path is optional; the
   default file name is `trace.jsonl`):

   ```bash
   ./bin/ttlang-sim examples/eltwise_add.py --trace /tmp/my_run.jsonl
   ```

2. **Print statistics from that file**:

   ```bash
   ./bin/ttlang-sim-stats /tmp/my_run.jsonl
   ```

Statistics are derived from trace events such as `copy_end`, `pipe_send`,
`pipe_recv`, `dfb_reserve_end`, and `dfb_wait_end`. If the trace was recorded
with a restricted event set, some tables may be empty. Regenerate the trace
with `ttlang-sim SCRIPT.py --trace` and the default categories, or enable the relevant
groups via `--trace-events` (see the tracing guide in `docs/TRACING.md` in the
repository). For full CLI details:

```bash
./bin/ttlang-sim-stats --help
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
