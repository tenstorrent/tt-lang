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

The operation uses a `(N workers, M workers)` grid. Every worker computes one
`m_block_tiles x n_block_tiles` output block.

- Column zero performs the fabric all-gather once for each M block. This avoids
  repeating the same fabric transfer for every N worker.
- A local row PipeNet distributes gathered activation blocks across N workers.
- Row zero reads each weight block. A local column PipeNet distributes it
  across M workers.
- Every output block has exactly one writer.

The configuration requires at least two workers on both axes. Tile counts must
be divisible by their corresponding block counts.

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
- row-broadcast bias;
- BF16 and FP32 interfaces;
- bit-exact gathered-activation validation and PCC output validation.

Not yet implemented:

- ReLU, GELU, and SiLU epilogues;
- chunked N output;
- addcmul;
- SwiGLU;
- transpose selection;
- fabric link, worker, and channel-buffer controls;
- FSDP weight gather.
