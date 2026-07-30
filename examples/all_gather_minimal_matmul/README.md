# All-Gather Minimal Matmul Fabric Examples

These examples model the core data dependence of the tt-metal implementation at
`third-party/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/`
`all_gather_minimal_matmul_async`.

The examples shard activations across K, all-gather activation shards across a
fabric mesh, shard weights across output N, and compute each device's local N
output shard. They do not model the full tt-metal operation surface: optional
bias, fused activation, addcmul, SwiGLU, chunked outputs, FSDP weight gather,
and program-factory core-grid scheduling are not included.

Files:

* `direct.py`: Direct DFB syntax. This is the primary example.
* `context.py`: Equivalent operation using context-manager DFB syntax.
* `common.py`: CLI parsing, mesh discovery, tensor setup, and correctness
  checks shared by both examples.

The TT-Lang operation does not allocate explicit tt-metal semaphores. Dataflow
buffer wait/reserve ordering plus PipeNet lowering provide the producer/consumer
synchronization that the explicit tt-metal program expresses through its runtime
semaphore protocol.

## Demo Checklist

- [x] **Baseline:** K-sharded activation all-gather, N-sharded matmul, direct
  and context-manager variants, bit-exact gather validation, and FP32 PyTorch
  matmul golden validation on four devices.
- [ ] **Item 1 - row-broadcast bias:** Add one N-sharded bias input and validate
  `A @ B + bias`.
- [ ] **Item 2a - ReLU:** Add an optional ReLU compute epilogue after bias.
- [ ] **Item 2b - GELU and SiLU:** Extend the validated activation interface
  after ReLU.
- [ ] **Item 3 - chunked N output:** Add output slicing and storage without
  changing the activation collective.
- [ ] **Item 4 - addcmul:** Add the full-output residual, row-broadcast
  multiplier, and scalar.
- [ ] **Item 5 - SwiGLU:** Add paired gate/up weight interpretation and
  half-width output validation.
- [ ] **Item 6a - multi-core scheduling:** Add configurable block, subblock,
  and compute-grid placement.
- [ ] **Item 6b - transpose selection:** Add the tt-metal transpose strategy
  and validate both selections.
- [ ] **Item 6c - fabric worker configuration:** Add link, worker-per-link, and
  channel-buffer controls.
- [ ] **Item 7 - FSDP weight gather:** Add the second collective and its
  independent synchronization domain.
- [ ] **Galaxy validation:** Validate the completed configuration on a
  32-device `4x8` mesh.

Each completed item receives a working checkpoint commit after direct and
context-manager correctness validation. Commit messages use the prefix
`[demo item N]`; the baseline uses `[demo baseline]`.

## Correctness Validation

The selected seed initializes one bf16 activation tensor and one bf16 weight
tensor on the host. The weights are scaled by `1 / K`. The same tensors are
sharded and transferred to the devices.

The golden output is the FP32 PyTorch matmul:

```python
expected_output = host_tensors.activation.float() @ host_tensors.weight.float()
```

The per-device N output shards are concatenated and compared with the golden
output using PCC. The default threshold is `0.99`. Independently, every gathered
activation shard on every destination device must be bit-exact with the
corresponding source slice of the original bf16 activation tensor.

## How to Run

The commands run from the repository root in an activated fabric-enabled
TT-Lang environment. They are identical in a local Docker container and an
Exabox container.

The direct DFB variant is the primary launcher:

```bash
python examples/all_gather_minimal_matmul_fabric_direct.py
```

The context-manager variant accepts the same arguments:

```bash
python examples/all_gather_minimal_matmul_fabric.py
```

Run on four devices:

```bash
set -o pipefail
timeout 300 python examples/all_gather_minimal_matmul_fabric_direct.py \
    --mesh-shape 2x2 \
    --m-tiles 1 \
    --k-tiles-per-device 1 \
    --k-tiles-per-transfer 1 \
    --n-tiles-per-device 1 \
    --fabric-config 2d \
    2>&1 | tee /tmp/device_test.log
```

Run on a 32-device Galaxy:

```bash
set -o pipefail
timeout 1800 python examples/all_gather_minimal_matmul_fabric_direct.py \
    --mesh-shape 4x8 \
    --m-tiles 4 \
    --k-tiles-per-device 8 \
    --k-tiles-per-transfer 1 \
    --n-tiles-per-device 8 \
    --fabric-config 2d \
    2>&1 | tee /tmp/device_test.log
```

Default execution without `--mesh-shape` uses `SystemMeshDescriptor` fabric
discovery. If the detected mesh has more than 32 devices, the example selects
the largest rectangular submesh that fits the current receive DFB slot limit.
If fabric discovery is unavailable, the fallback selects a factorized mesh from
the available device count, using `4x8` for 32 devices.

`--fabric-reliability-mode` controls the topology-health policy passed to
`ttnn.initialize_mesh_device_fabric`:

* `relaxed` maps to `RELAXED_INIT`. Fabric initialization accepts unavailable
  links or devices and uses the routing planes supported by the live topology.
  This is the default.
* `strict` maps to `STRICT_INIT`. Live links and devices must exactly match the
  mesh graph descriptor; unavailable hardware causes initialization to fail.
* `dynamic` maps to `DYNAMIC_RECONFIG`. This is a tt-metal placeholder for
  runtime response to hardware failures and is currently unsupported.

The option affects fabric initialization only. It does not change the transfer
relation, matmul computation, or correctness checks.

On shared hosts, writable tt-metal cache directories avoid failures caused by
cache entries owned by another user:

```bash
export TT_METAL_CACHE=/tmp/tt-metal-cache-$USER
export TT_METAL_LOGS_PATH=/tmp/tt-metal-logs-$USER
```
