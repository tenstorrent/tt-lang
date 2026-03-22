# Implementation Notes

This section collects design documents and pipeline traces that describe how tt-lang lowers kernels from Python to hardware code. These are intended for contributors and anyone who needs to understand compiler internals.

## Design Documents

- [DST Register Allocation](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/DST_Allocation.md) — how the `TTLAssignDST` pass assigns destination registers to tile operations
- [DST Register Utilization](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/DST_Utilization.md) — maximizing tile throughput per DST synchronization cycle

## Lowering Pipeline Traces

These documents trace specific operations through the full compiler pipeline, from Python input through MLIR passes to generated C++ kernel code.

- [Multi-tile Compute Operations](https://github.com/tenstorrent/tt-lang/blob/main/docs/LOWERING_MULTITILE.md) — traces a 2x2 multi-tile add through the pipeline
