# Programming Guide

This page covers compiler options, print debugging, performance tools, the simulator, and examples for tt-lang kernel development.

## Compiler Options

Kernels accept compiler options that control code generation (e.g., `--no-ttl-maximize-dst`, `--no-ttl-fpu-binary-ops`). These can be passed as command-line arguments, via the `@ttl.kernel` decorator's `options=` parameter, or the `TTLANG_COMPILER_OPTIONS` environment variable. Command-line arguments take highest priority.

```bash
# List available options
python examples/tutorial/multicore_grid_auto.py --ttl-help

# Run a kernel with options
python examples/tutorial/multicore_grid_auto.py --no-ttl-maximize-dst
```

See `python/ttl/compiler_options.py` for details on priority ordering and the merge protocol.

## Print Debugging

Use `print()` inside kernel code to emit device debug prints. Enable at runtime with `TT_METAL_DPRINT_CORES`:

```bash
export TT_METAL_DPRINT_CORES=0,0   # core to capture
python my_kernel.py 2>&1 > output.txt
```

```python
@ttl.compute()
def compute():
    with inp_dfb.wait() as tile, out_dfb.reserve() as o:
        print("hello")                             # auto: math thread
        print(tile)                                # auto: pack thread
        result = ttl.exp(tile)
        print(_dump_dst_registers=True, label="after exp") # auto: math thread
        o.store(result)

@ttl.datamovement()
def dm_write():
    print(out_dfb)                               # CB metadata
    with out_dfb.wait() as blk:
        print(blk, num_pages=1)                  # raw tensor page
        tx = ttl.copy(blk, out[0, 0])
        tx.wait()
```

- Prints can be extremely large and slow; redirect output to a file and use grep.
- In compute kernels, guard prints with `thread="math"`, `thread="pack"`, or `thread="unpack"` to avoid overlapping output from the three TRISC threads.
- When using multi-tile block sizes (CB shape > 1x1), prints inside the generated loop will dump all tiles in the block.

See the [full print debugging reference](reference/print-debugging.md) for all supported modes (scalars, tiles, tensor pages, CB details, DST registers, thread conditioning).

## Performance Tools

TT-Lang includes built-in performance analysis tools for profiling kernels on hardware:

- Perf Summary (`TTLANG_PERF_DUMP=1`) — NOC traffic and per-thread wall time breakdown
- Auto-Profiling (`TTLANG_AUTO_PROFILE=1`) — automatic per-line cycle count instrumentation
- User-Defined Signposts (`TTLANG_SIGNPOST_PROFILE=1`) — targeted cycle counts for `ttl.signpost()` regions
- Perfetto Trace Server (`TTLANG_PERF_SERV=1`) — visualize profiler data in the Perfetto UI

Performance tracing (Tracy) is enabled by default at build time. To disable it, configure with `-DTTLANG_ENABLE_PERF_TRACE=OFF`.

See the [full performance tools reference](reference/performance-tools.md) for environment variable details, valid combinations, and sample output.

## Simulator

For users who want to run simulator examples without building the full compiler stack:

```bash
./bin/ttlang-sim examples/eltwise_add.py
python -m pytest test/sim/
```

The simulator runs as standard Python code, enabling any Python debugger to work with it.

### VSCode Debugger

Create a debug configuration in `.vscode/launch.json`:

```json
{
  "name": "Debug TTL Simulator",
  "type": "debugpy",
  "request": "launch",
  "module": "sim.ttlang_sim",
  "args": ["${file}"],
  "console": "integratedTerminal",
  "justMyCode": false,
  "cwd": "${workspaceFolder}",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/python"
  }
}
```

1. Open a kernel file in VSCode (e.g., `examples/eltwise_add.py`)
2. Set breakpoints in your kernel code
3. Press F5 or select "Debug TTL Simulator" from the Run menu
4. The debugger stops at breakpoints, allowing variable inspection and step-through execution

## Examples

See the `examples/` and `test/` directories for complete working examples, including:
- `test/python/simple_add.py`
- `test/python/simple_fused.py`

The [tutorial](ttl-tutorial/index.md) provides step-by-step examples from single-tile to multinode kernels.
