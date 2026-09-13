# All-gather minimal matmul

This example computes `all_gather_K(activation) @ weight + bias` on a device
line. Activation is K-sharded; weight, bias, and output are N-sharded. The four
output shards collectively contain one `M x N` result.

| Order | Implementation | Activation exchange | Status |
| ---: | --- | --- | --- |
| 1 | [`operation.py`](operation.py) | One M block per fabric transfer; direct L1 injection | Correct; 2.613 ms control |
| 2 | [`operation_bidirectional_dram.py`](operation_bidirectional_dram.py) | Bidirectional K halves staged in receiver DRAM | Correct; slower than direct L1 |
| 3 | [`operation_grouped_rows.py`](operation_grouped_rows.py) | Three contiguous M blocks per fabric transfer; L1 subview injection | Selected; 2.306 ms |

## Dataflow

```text
for each M block:
    forward each device's activation K shard around the device ring
    forward local and received activation blocks along its compute chain
    multicast each N-sharded weight block down its compute column
    initialize the FP32 accumulator from the local bias shard
    accumulate every global K block while communication continues
    convert once to the output dtype and write the local N shard to DRAM
```

Four workers per device communicate over fabric. Each transfers the activation
data for three compute rows, then injects an L1 subview into each of the 12
compute chains. Each block advances through the ten compute nodes in its chain.
The `12 x 10` compute grid uses 120 workers per device.
Bounded dataflow buffers provide backpressure between data movement and compute;
the operation does not allocate gathered-activation DRAM.

## Run

Small four-device case:

```bash
python -m examples.all_gather_minimal_matmul --mesh-shape 4x1
```

Selected Wan2.2 QKV benchmark, with per-device `M/K/N=9472/1280/3840`:

```bash
python -m benchmarks.all_gather_minimal_matmul \
    --implementation ttlang \
    --ttlang-activation-strategy grouped-row-l1 \
    --mesh-shape 4x1 \
    --ttlang-compute-grid 12 10 \
    --ttlang-communication-workers 4 \
    --m-tiles 296 \
    --k-tiles-per-device 40 \
    --n-tiles 480 \
    --ttlang-m-block-tiles 5 \
    --ttlang-k-block-tiles 10 \
    --ttlang-n-block-tiles 12 \
    --no-ttlang-reuse-activation
```

The compiler prints generated C++ filenames under `/tmp/default` in the
container. The three kernels are `receive_activations_and_write_output`,
`forward_activations_and_distribute_weights`, and `compute_matmul_and_bias`.

[`config.py`](config.py) validates the static device, worker, and tile
decomposition. The [benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
compares the example with TT-Metal's fused operation.
