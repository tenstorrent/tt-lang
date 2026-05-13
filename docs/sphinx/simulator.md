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

## Float32 Promotion

By default the simulator promotes all floating-point dtypes narrower than
float32 to float32 before any computation:

| Declared dtype | Simulator dtype |
|---|---|
| `ttnn.bfloat16` | `torch.float32` |
| `ttnn.float16` | `torch.float32` |
| `ttnn.bfloat8_b` | backed by `torch.float32` |
| `ttnn.float32` | `torch.float32` (unchanged) |

This makes the simulator work correctly on host architectures that lack native
support for narrow float types (e.g. Apple Silicon has no hardware bfloat16 or
float16 support, so using those types natively would be slow or incorrect).

### Disabling promotion

Pass `--no-float32-promotion` to `ttlang-sim` to run with the dtypes declared
in the source file:

```bash
ttlang-sim --no-float32-promotion examples/matmul_1d.py
```

### When to disable promotion

**Correctness checks calibrated for the original dtype.** Examples that use
ULP-based assertions (`assert_with_ulp`) with tolerances chosen for bfloat16
precision will fail when run in float32, because the same absolute numerical
difference corresponds to more ULPs in float32 (which has a smaller ULP than
bfloat16). Run these with `--no-float32-promotion`:

- `examples/matmul_1d.py`
- `examples/matmul_1d_mcast.py`
- `examples/metal_examples/single_node_matmul/ttlang/single_node_matmul.py`
- `examples/metal_examples/multinode_matmul/ttlang/multinode_matmul.py`

**L1 memory budget.** The simulator uses the declared dtype for all
`DataflowBuffer` capacity accounting so the reported footprint always matches
what the hardware would allocate, regardless of whether float32 promotion is
active. If the total buffer capacity for a core exceeds the L1 limit, the
simulator issues a warning:

```
UserWarning: Total DataflowBuffer capacity per core (N bytes) exceeds the L1 memory limit of M bytes.
Memory is accounted using declared dtypes, so this reflects the on-hardware footprint of the kernel.
```

This warning does not abort execution, but it indicates that the kernel would
not fit in hardware L1.

**Dtype-specific behavior.** If a kernel explicitly tests dtype identity,
overflow behavior, or precision characteristics of a specific narrow type,
disable promotion so the script runs with the declared dtype.

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
