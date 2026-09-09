# All-Gather Minimal Matmul

This package models the data dependence of TT-Metal's
`all_gather_minimal_matmul_async` at the TT-Metal revision pinned by this
repository:

`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async`

The operation accepts:

- an activation tensor sharded across K;
- a weight tensor sharded across output N;
- a row-broadcast bias sharded across output N;
- a replicated activation output used to validate the collective;
- an output tensor sharded across N.

Each device gathers every activation K shard, computes its local N output, and
adds its local bias. Passing a zero bias tensor selects the unbiased result.
Inputs and outputs currently support TILE layout in DRAM with BF16 or FP32
elements.

[`operation.py`](operation.py) contains the configuration and fused operation;
[`__main__.py`](__main__.py) is the standalone correctness driver;
[`__init__.py`](__init__.py) exports the public configuration and factory.

## Worker decomposition

The default grid is `(N workers, M workers)`; `transpose=True` exchanges the
physical axes. An explicit `worker_grid` distributes successive M/N blocks
cyclically across a fixed set of workers. Each worker computes
`m_block_tiles x n_block_tiles` blocks, with M rounds outside N rounds.

- Column zero performs the fabric all-gather once for each M block. This avoids
  repeating the same fabric transfer for every N worker.
- A local row PipeNet distributes gathered activation blocks across N workers.
- Row zero reads each weight block. A local column PipeNet distributes it
  across M workers.
- Every output block has exactly one writer.

`k_tiles_per_transfer` controls fabric messages; `k_block_tiles` independently
controls matmul operand blocks and defaults to the transfer size. Both must
divide the per-device K extent. During the first N round, each common group of
`lcm(k_tiles_per_transfer, k_block_tiles)` K tiles is gathered into DRAM and
published to compute before gathering the next group. Subsequent N rounds reuse
that gathered data. Both operand streams use group/block/device order, while
gathered output retains canonical device order.

Sequential send/local/receive staging uses one DFB block each; operand DFBs
remain double-buffered. The activation reader is declared first to select
NoC 0, with weight traffic on NoC 1. This assignment is measured on the
transposed Blackhole workload; it is not a claim of optimal placement for
every grid or architecture.

The configuration requires at least two workers on both axes. Tile counts must
be divisible by their block extents, and M/N block counts must be divisible by
the corresponding worker counts. Edge blocks are not padded implicitly.
The runtime additionally checks available fabric connections and L1 capacity;
the [benchmark comparison table](../../benchmarks/all_gather_minimal_matmul/README.md#comparison-with-the-native-benchmark)
records the measured local limits.

## Run

Run from the repository root in a fabric-enabled TT-Lang environment:

```bash
set -o pipefail
timeout 300 python -m examples.all_gather_minimal_matmul \
    --mesh-shape 1x2 \
    --m-tiles 2 \
    --k-tiles-per-device 1 \
    --n-tiles-per-device 2 \
    2>&1 | tee /tmp/device_test.log
```

Omit `--mesh-shape` to use the control-plane-discovered mesh. `--dtype` accepts
`bf16` and `fp32`; `--no-bias` supplies a zero bias tensor.

The [local benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
compares this operation with the original TT-Metal fused operation, including
correctness checks and TT-Metal device-kernel profiling.

## TT-Metal feature correspondence

Implemented:

- K-sharded activation all-gather;
- N-sharded matmul output;
- full-grid M/N worker decomposition;
- transposed scheduling and repeated output blocks on a fixed worker grid;
- row-broadcast bias;
- BF16 and FP32 interfaces;
- bit-exact gathered-activation validation and PCC output validation.

Not yet implemented:

- ReLU, GELU, and SiLU epilogues;
- chunked N output;
- addcmul;
- SwiGLU;
- partially occupied worker grids and edge blocks;
- fabric link, worker, and channel-buffer controls;
- FSDP weight gather.
