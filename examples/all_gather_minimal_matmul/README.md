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

Default execution uses `SystemMeshDescriptor` fabric discovery. If the detected
mesh has more than 32 devices, the example selects the largest rectangular
submesh that fits the current receive DFB slot limit. If fabric discovery is
unavailable, the fallback selects a factorized mesh from the available device
count, using `4x8` for 32-device Galaxy systems. Override this with
`--mesh-shape ROWSxCOLS` when a specific topology descriptor is required.
The default fabric reliability mode is `relaxed`, which matches current Galaxy
bring-up expectations when the physical topology has extra or degraded links.
Use `--fabric-reliability-mode strict` when exact descriptor matching is
required.

Run from the repository root:

```bash
python examples/all_gather_minimal_matmul_fabric_direct.py
python examples/all_gather_minimal_matmul_fabric.py
```

## Local Four-Device QB

The dated branch `bnorris/demo-local-4-device-20260730` is the pinned local
demo revision. The validated checkout is
`/home/bnorris/tt/tt-lang4-local-demo-20260730`, and the validated container is
`bnorris-ird-fabric-v1.1.7`.

Build the dated branch:

```bash
cd /home/bnorris/tt/tt-lang4-local-demo-20260730
docker exec -w "$PWD" bnorris-ird-fabric-v1.1.7 \
  bash -c 'source build-docker/env/activate && cmake --build build-docker'
```

Before each run, verify that no process owns a device and reset all four
devices. The wrapper preserves the complete device visibility needed for
`tt-smi -r all`.

```bash
fuser -v /dev/tenstorrent/*
/home/bnorris/soft/bin/tt-run-when-free \
  /home/bnorris/.local/bin/tt-smi -r all
```

Run the direct variant:

```bash
docker exec -w "$PWD" bnorris-ird-fabric-v1.1.7 \
  bash -c 'set -o pipefail;
    source build-docker/env/activate &&
    timeout 300 python examples/all_gather_minimal_matmul_fabric_direct.py \
      --mesh-shape 2x2 \
      --m-tiles 1 \
      --k-tiles-per-device 1 \
      --k-tiles-per-transfer 1 \
      --n-tiles-per-device 1 \
      --fabric-config 2d \
      --fabric-reliability-mode relaxed \
      2>&1 | tee /tmp/device_test.log'
```

The context-manager variant uses the same arguments:

```bash
docker exec -w "$PWD" bnorris-ird-fabric-v1.1.7 \
  bash -c 'set -o pipefail;
    source build-docker/env/activate &&
    timeout 300 python examples/all_gather_minimal_matmul_fabric.py \
      --mesh-shape 2x2 \
      --m-tiles 1 \
      --k-tiles-per-device 1 \
      --k-tiles-per-transfer 1 \
      --n-tiles-per-device 1 \
      --fabric-config 2d \
      --fabric-reliability-mode relaxed \
      2>&1 | tee /tmp/device_test.log'
```

Both commands use all four devices. A successful run reports a PCC above the
configured threshold and exits with status zero. The validation on 2026-07-30
reported PCC `0.999997` for each variant.

Run on the largest discovered mesh containing at most four devices:

```bash
python examples/all_gather_minimal_matmul_fabric_direct.py \
  --mesh-shape auto \
  --max-devices 4 \
  --m-tiles 1 \
  --k-tiles-per-device 1 \
  --k-tiles-per-transfer 1 \
  --n-tiles-per-device 1 \
  --fabric-reliability-mode relaxed
```

Run on the largest discovered mesh containing at most 32 devices:

```bash
python examples/all_gather_minimal_matmul_fabric_direct.py \
  --mesh-shape auto \
  --max-devices 32 \
  --m-tiles 4 \
  --k-tiles-per-device 8 \
  --k-tiles-per-transfer 1 \
  --n-tiles-per-device 8 \
  --fabric-reliability-mode relaxed
```

On shared Exabox hosts, use a writable tt-metal cache. Stale root-owned cache
entries can prevent fabric router kernels from being generated:

```bash
export TT_METAL_CACHE=/tmp/tt-metal-cache-$USER
export TT_METAL_LOGS_PATH=/tmp/tt-metal-logs-$USER
```
