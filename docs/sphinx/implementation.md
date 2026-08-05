# Implementation Notes

This section collects design documents and pipeline traces that describe how TT-Lang lowers operations from Python to hardware code. These are intended for contributors and anyone who needs to understand compiler internals.

## Design Documents

- [Static Execution Analysis](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/StaticExecutionAnalysis.md) - exact operation cardinality in structured control flow
- [Value Origin Analysis](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/ValueOriginAnalysis.md) - conservative SSA value origins through control flow and tensor updates
- [`ComputeOp` Creation and Fusion](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/ComputeOpCreation.md) - immutable planning for `ttl.compute` creation, fusion, DFB publication, and intermediate materialization
- [DST Register Allocation](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/DST_Allocation.md) — how the `TTLAssignDST` pass assigns destination registers to tile operations
- [DST Register Utilization](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/DST_Utilization.md) — maximizing tile throughput per DST synchronization cycle
- [External Function Interop Lowering](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/ExternalFuncInteropLowering.md) - typed DFB descriptors and argument mapping for `ttl.call_extern_func`

## Lowering Pipeline Traces

These documents trace specific operations through the full compiler pipeline, from Python input through MLIR passes to generated C++ kernel code.

- [Multi-tile Compute Operations](https://github.com/tenstorrent/tt-lang/blob/main/docs/LOWERING_MULTITILE.md) — traces a 2x2 multi-tile add through the pipeline
