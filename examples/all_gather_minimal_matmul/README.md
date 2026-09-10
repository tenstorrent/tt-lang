# All-Gather Minimal Matmul

Four-device M/K/N=9472/5120/15360 device times: replicated TT-Lang 10.343 ms
(130 compute workers/device), N-sharded plus output gather 24.156 ms (60),
and native 6.883 ms (108). See the [performance report](../../benchmarks/all_gather_minimal_matmul/PERFORMANCE.md).

| Entry point | Result on each device |
| --- | --- |
| [`n_sharded/`](n_sharded/) | `M x N/D`; add `--gather-output` for replicated `M x N`. |
| [`replicated/`](replicated/) | Replicated `M x N`, computed independently with replicated weights and bias. |

`N` is the complete output width and `D` is the device count. Weights and bias
remain N-sharded when output gathering is enabled. The final gather copies
output columns into device order without arithmetic.
`--activation-all-gather` and `--output-all-gather` independently select
`all_to_all` or `ring`. Ring forwarding uses L1 DFBs. Inside matmul it changes
K-block accumulation order; correctness uses dtype-specific tolerances.

```bash
python -m examples.all_gather_minimal_matmul.n_sharded --mesh-shape 2x2 --gather-output
python -m examples.all_gather_minimal_matmul.replicated --mesh-shape 2x2 --n-tiles 8
```

[`collectives.py`](collectives.py) implements activation and output gathering.
[`operation.py`](operation.py) selects the N-sharded fused operation:
[`per_row_all_gather/operation.py`](per_row_all_gather/operation.py) assigns one
communication worker to each M-worker row;
[`two_worker_ring/operation.py`](two_worker_ring/operation.py) assigns two
communication workers to all rows when ring forwarding uses more than two M workers.
[`replicated/operation.py`](replicated/operation.py) composes activation gathering
with replicated matmul.
The [shared benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
includes the final gather in device timing and reports four-device results
for both output strategies.

This package models the data dependence of TT-Metal's
`all_gather_minimal_matmul_async` at the TT-Metal revision pinned by this
repository:

`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async`

The operation accepts:

- an activation tensor sharded across K;
- a weight tensor either N-sharded or replicated;
- row bias with the same N placement as the weights;
- output with the same N placement as the weights, optionally gathered afterward.

Each device gathers every activation K shard, computes its local N output, and
adds its local bias. Passing a zero bias tensor selects the unbiased result.
Inputs and outputs currently support TILE layout in DRAM with BF16 or FP32
elements.

[`config.py`](config.py) contains the tensor blocking and worker configuration;
[`__main__.py`](__main__.py) is the standalone correctness driver;
[`__init__.py`](__init__.py) exports the public configuration and factory.

## Replicated-output execution

The replicated variant first gathers activation shards into a complete DRAM
tensor, then runs matmul independently on every device. The caller supplies
the gathered-activation storage; there is no final output gather. On one device,
matmul reads the input directly and no collective or scratch tensor is needed.

Matmul multicasts activation blocks across N workers and weights across M workers.
Bias initializes the L1 accumulator before K accumulation. Streaming receives
directly into the compute DFB; full-K caching retains separate receive storage
so republishing cached pages does not change multicast receiver addresses.
The collective and matmul currently execute sequentially.

## N-sharded worker decomposition

The default grid is `(N workers, M workers)`; `transpose=True` exchanges the
physical axes. An explicit `worker_grid` distributes successive M/N blocks
cyclically across a fixed set of workers. Each worker computes
`m_block_tiles x n_block_tiles` blocks, with M rounds outside N rounds.

- Column zero performs the fabric all-gather for each M block, repeating it
  across N rounds only when activation reuse is disabled.
- A local row PipeNet distributes gathered activation blocks across N workers.
- Row zero reads each weight block. A local column PipeNet distributes it
  across M workers.
- Every output block has exactly one writer.

`k_block_tiles` controls both fabric messages and matmul operand blocks and
must divide the per-device K extent. The first N round receives remote blocks
into L1, interleaves them with local blocks in K-block/device order, and
broadcasts them across each worker row. No gathered activation tensor or DRAM
scratch is allocated or written.

By default, `reuse_activation=True` retains one M block's full K extent in each
worker's activation DFB. Subsequent N rounds republish the same L1 pages without
fabric transfers, DRAM reads or row broadcasts. Full-K DFB capacity and matching
producer/consumer order preserve the cached pages until the next M block.
`reuse_activation=False` instead streams every N round through a two-block DFB;
it uses less L1 but repeats activation communication. Reuse supports at most
32 K blocks; larger reductions require larger K blocks or streamed activations.
Actual capacity is also limited by per-worker L1 allocation.

Direct fabric receive holds one block per remote device; ring receive and
forwarding use two-block DFBs. Separate row-receive storage preserves the
multicast protocol's receiver address sequence while cached compute pages are
republished. Weight operands remain double-buffered. FP32 destinations select
FP32 packer accumulation directly from BF16 or FP32 matmul operands, avoiding
intermediate BF16 rounding and SFPU accumulation. Bias is converted and added
after the reduction, followed by output conversion.

The activation reader is declared first to select NoC 0, with weight traffic
on NoC 1. This assignment is measured on the
transposed Blackhole workload; it is not a claim of optimal placement for
every grid or architecture.

The configuration requires at least two workers on both axes. Tile counts must
be divisible by their block extents, and N block counts must be divisible by
the N worker count. The runner pads activation rows to `config.padded_m_tiles * 32`
before allocation; the operation writes only the original M rows. Direct callers
must provide that padded activation storage.
With ring all-gather and more than two M workers, two communication workers
serve all compute rows through NoC multicast. Each communication worker stages
one block per assigned row in L1. This supports a 13x10 compute grid without
additional fabric connections; it does not imply better performance than a
smaller grid.
The runtime additionally checks available fabric connections and L1 capacity;
the [benchmark comparison table](../../benchmarks/all_gather_minimal_matmul/README.md#comparison-with-the-native-benchmark)
records the tested hardware and configuration limits.

## Run

Run from the repository root in a fabric-enabled TT-Lang environment:

```bash
set -o pipefail
timeout 300 python -m examples.all_gather_minimal_matmul \
    --mesh-shape 2x2 \
    --m-tiles 2 \
    --k-tiles-per-device 1 \
    --n-tiles-per-device 2 \
    2>&1 | tee /tmp/device_test.log
```

Omit `--mesh-shape` to use the control-plane-discovered mesh. `--dtype` accepts
`bf16` and `fp32`; `--no-bias` supplies a zero bias tensor. Set
`--mesh-shape 1x1` to run the same operation with an identity all-gather and
fabric disabled.

130-worker, four-device example (BF16, TILE/DRAM):

```bash
python -m examples.all_gather_minimal_matmul.n_sharded \
    --mesh-shape 2x2 --worker-grid 13 10 --transpose \
    --m-tiles 24 --k-tiles-per-device 2 --n-tiles 80 \
    --activation-all-gather ring --gather-output --output-gather-workers 2
```

Matmul uses 130 workers per device; the subsequent output gather uses two.

The [device-time benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
compares this operation with the original TT-Metal fused operation, including
correctness checks and TT-Metal device-kernel profiling.

## TT-Metal feature correspondence

| Capability | TT-Lang status |
| --- | --- |
| K-sharded activation all-gather | Direct transfers and ring forwarding. |
| N-sharded weights and output | Implemented; optional final gather replicates the result. |
| Replicated weights, bias and output | Implemented. |
| Worker decomposition | Fixed M/N grid, optional transpose, repeated output blocks. |
| Arithmetic | BF16/FP32 TILE tensors, row bias, FP32 packer accumulation. |
| Activation storage | Full-K L1 reuse or two-block streaming; bounded-K reuse remains to be implemented. |
| Partially occupied M grids | Zero-padded activation rows; stores restricted to logical M. Tile/block alignment remains required. |
| Exact/approximate GELU | Not implemented. |
| Two-/three-way and unequal-width N splits | Not implemented. |
| Scaled addcmul with row/full multipliers | Not implemented. |
| SwiGLU | Not implemented. |
| FSDP weight gather | Not implemented. |
| Fabric link/worker/channel-buffer controls | Not exposed by this example. |

The upstream sweep also covers separate operations: standalone/split matmul,
strided all-gather matmul, and matmul + reduce-scatter + addcmul. These are not
implemented by this example.
