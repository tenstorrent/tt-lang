# All-gather minimal matmul

This example computes `all_gather_K(activation) @ weight + bias` on a device
line. Activation is K-sharded; weight, bias, and output are N-sharded. The four
output shards collectively contain one `M x N` result.

| Order | Implementation | Inter-device communication | Status |
| ---: | --- | --- | --- |
| 1 | [`operation.py`](operation.py) | One M block per fabric transfer; direct L1 injection | Correct; 2.613 ms control |
| 2 | [`operation_bidirectional_dram.py`](operation_bidirectional_dram.py) | Bidirectional K halves staged in receiver DRAM | Correct; slower than direct L1 |
| 3 | [`operation_grouped_rows.py`](operation_grouped_rows.py) | Three contiguous M blocks per fabric transfer; L1 subview injection | Correct; 2.269 ms |
| 4 | [`operation_bidirectional_l1.py`](operation_bidirectional_l1.py) | Bidirectional K halves received into L1 and distributed from opposite rows | Selected; 1.847 ms |
| 5 | [`../matmul_reduce_scatter_2d/operation.py`](../matmul_reduce_scatter_2d/operation.py) | Exchange partial results between two K groups and reduce into M/N-sharded output | Correct; 3.387 ms on four devices |

Comparison reference: TT-Metal
[`all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/tree/f8c4ce59dd04a3eeeb11abf01ffc9dbce0059eba/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async),
which returns the same N-sharded output; [benchmark commands](../../benchmarks/all_gather_minimal_matmul/README.md).

| Equivalent implementation | Physical source lines | Native/TT-Lang |
| --- | ---: | ---: |
| TT-Lang [`operation_bidirectional_l1.py`](operation_bidirectional_l1.py) | 490 | 1.0 |
| Native TT-Metal API, device operation, program factory, and device kernels | 5,905 | 12.1 |

Counts use `wc -l`; generated C++, bindings, tests, and documentation are
excluded.

## Selected column-parallel dataflow

```text
for each M block and local K block:
    split the activation block into left and right K halves
    stream the halves in opposite directions around the device ring
    distribute each received half from its boundary row across the compute grid
    distribute the matching N-sharded weight halves down each compute column
    initialize the FP32 accumulator from the local bias shard
    accumulate both halves for every source device while communication continues
    convert once to the output dtype
    publish the next inputs, then write the preceding local N shard to DRAM
```

Four compute rows provide the sender and receiver clients for both fabric
directions. Eight mux workers map the clients onto four links per direction.
The `12 x 10` compute grid uses 120 workers per device. Bounded dataflow buffers
provide backpressure; the operation does not allocate gathered-activation DRAM.

## Run

Small four-device case:

```bash
python -m examples.all_gather_minimal_matmul \
    --mesh-shape 4x1 \
    --activation-strategy bidirectional-l1 \
    --no-reuse-activation
```

Selected Wan2.2 QKV benchmark, with per-device `M/K/N=9472/1280/3840`:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang \
    --ttlang-activation-strategy bidirectional-l1 \
    --mesh-shape 4x1 \
    --ttlang-compute-grid 12 10 \
    --m-tiles 296 \
    --k-tiles-per-device 40 \
    --n-tiles 480 \
    --ttlang-m-block-tiles 5 \
    --ttlang-k-block-tiles 10 \
    --ttlang-n-block-tiles 12 \
    --no-ttlang-reuse-activation
```

The compiler prints generated C++ filenames under `/tmp/default` in the
container. The three kernels are `move_activations`,
`move_weights_and_write_output`, and `compute_matmul_and_bias`.

[`config.py`](config.py) validates the static device, worker, and tile
decomposition. The [benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
compares the example with TT-Metal's fused operation.

## Two-dimensional decomposition

The 2D operation uses a `P_K x P_N` device mesh. Activation is K-sharded and
replicated across `P_N`; weight and bias are K/N-sharded. Each device computes
one partial `M x N/P_N` result, exchanges it with the other K group, and retains
one `M/P_K x N/P_N` output shard. See
[`matmul_reduce_scatter_2d/operation.py`](../matmul_reduce_scatter_2d/operation.py).
