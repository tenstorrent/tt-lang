# All-gather minimal matmul

This example computes `all_gather_K(activation) @ weight + bias` on a device
line. Activation is K-sharded; weight, bias, and output are N-sharded. The four
output shards collectively contain one `M x N` result.

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

Four workers per device communicate over fabric and inject activation blocks
directly into the 12 compute chains. Each block then advances through the ten
compute nodes in its chain. The `12 x 10` compute grid uses 120 workers per device.
Bounded dataflow buffers provide backpressure between data movement and compute;
the operation does not allocate gathered-activation DRAM.

## Run

Small four-device case:

```bash
python -m examples.all_gather_minimal_matmul --mesh-shape 4x1
```

Wan2.2 QKV dimensions, with per-device `M/K/N=9472/1280/3840`:

```bash
python -m examples.all_gather_minimal_matmul \
    --mesh-shape 4x1 \
    --compute-grid 12 10 \
    --communication-workers 4 \
    --m-tiles 296 \
    --k-tiles-per-device 40 \
    --n-tiles 480 \
    --m-block-tiles 5 \
    --k-block-tiles 10 \
    --n-block-tiles 12 \
    --no-reuse-activation
```

The compiler prints generated C++ filenames under `/tmp/default` in the
container. The three kernels are `receive_activations_and_write_output`,
`forward_activations_and_distribute_weights`, and `compute_matmul_and_bias`.

[`operation.py`](operation.py) contains the TT-Lang implementation.
[`config.py`](config.py) validates the static device, worker, and tile
decomposition. The [benchmark](../../benchmarks/all_gather_minimal_matmul/README.md)
compares the example with TT-Metal's fused operation.
