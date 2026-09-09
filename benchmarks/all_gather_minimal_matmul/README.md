# Local all-gather matmul comparison

Compares the TT-Lang example with the installed TT-Metal
[`ttnn.experimental.all_gather_minimal_matmul_async`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.cpp), using TT-Metal's standard
device profiler and `device_kernel_duration` analysis.

The source permalink uses the repository dependency pin; it does not establish
the build revision of the installed native binary.

## Files

- [`__main__.py`](__main__.py): CLI, matched workload construction, correctness
  checks, isolated profiler workers, and JSON results/provenance.
- [`profile.py`](profile.py): optional cProfile diagnosis of Python/runtime
  overhead. Its host timings are not the kernel comparison.
- [`../device_timing.py`](../device_timing.py): adapter to TT-Metal's
  `tracy.process_device_log.import_log_run_stats`; no local CSV parser.
- [PERFORMANCE.md](PERFORMANCE.md): measurements and generated/native C++ analysis.
  Detailed reports and source snapshots are kept in the
  [external artifact archive](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088), not in the repository.
- [`__init__.py`](__init__.py): package marker, not a kernel.
- [`operation.py`](../../examples/all_gather_minimal_matmul/operation.py):
  TT-Lang all-gather + matmul + bias implementation, shared with the
  [example](../../examples/all_gather_minimal_matmul/README.md) and
  [device pytest](../../test/python/fabric/test_all_gather_minimal_matmul.py).
  Native kernels come from the installed `ttnn` package.

The [matmul directory](../matmul/README.md) contains the single-device
SUMMA/K-split comparison.

## Comparison with the native benchmark

The reference is TT-Metal's
[block-size sweep](https://github.com/tenstorrent/tt-metal/blob/f69f924c6b4f38daa0a6f25716731f36c573dc0e/models/tt_dit/utils/sweep_mm_block_sizes.py)
and [Wan2.2 operation test](https://github.com/tenstorrent/tt-metal/blob/f69f924c6b4f38daa0a6f25716731f36c573dc0e/models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py).
These describe the benchmark methodology, not the installed binary revision.
The local comparison applies identical workload, arithmetic, worker grid and
M/K/N blocks to both implementations. It is not an exact Galaxy reproduction.

| Setting | Native Blackhole reference | Local comparison | Difference and consequence |
| --- | --- | --- | --- |
| Problem dimensions | Plain bias tests include M=3072, full K=5120, per-device N=1280 or 3840. | CLI supports these dimensions through repeated output blocks; default remains M=64/K=64/per-device N=128. | Default is a smoke test, not a model-performance result. N in the reference is per device, not global output width. |
| Participants | Default sweep opens a 4x8 parent and uses a 4x1 ring; alternative uses a 1x8 ring. | Discovered 2x2 parent, connected 2x1 submesh, physical IDs recorded per run. | Two participants, not four/eight; local K=2560 for full K=5120, versus K=1280 on four participants. |
| Worker scheduling | 12x9, transposed; multiple blocks and edge blocks. | Explicit fixed `--worker-grid X Y` and `--transpose`; repeated M/N blocks. | Block counts must divide worker extents. A 12x5 configuration passes that condition but fails local fabric assignment: generated per-worker send/receive managers require distinct forwarding connections, with only four eligible links. Two M workers work locally. Full 12x9 edge scheduling and native mux sharing are not implemented. |
| M/K/N blocks | Plain M=3072 cases use 8/8/8 tiles in the test; sweep tunes blocks. | Configurable and identical between local implementations. Initial measurements use 2/2/2; tuning selects 2/40/2 with the same 2x5 grid. `--k-block-tiles` sets compute K independently of TT-Lang's fabric transfer size. | Larger K blocks amortize matmul initialization, intermediate packing, local multicast synchronization and DRAM barriers. Initial 8/8/8 exceeded L1; the historical 4/8/8 attempt timed out. |
| Arithmetic | BF16, HiFi2, FP32 destinations, packer L1 accumulation. | BF16 uses HiFi2; the separate FP32 variant uses HiFi4. FP32 destinations and explicit FP32 accumulator/bias DFBs by default. | BF16 matmul-block products are packed to BF16 before conversion and FP32 addition; native uses packer L1 accumulation. Destination precision matches, but intermediate rounding and storage do not. FP32 HiFi2 fails the FP32 error bound; HiFi4 is applied to both implementations. |
| Subblocks | Test uses 2x2 for the plain M=3072 cases; sweep selects valid subblocks. | Native uses 2 on each evenly divisible block axis, otherwise 1. TT-Lang compiler-selected; a 2x2 output block fits four FP32 DST tiles. | Larger block configurations can still have different register scheduling. |
| Fabric | Ring, 2 links, 6 workers/link; sweep uses 24 channel buffers, operation test uses 48. | Linear, one link; native workers/link equals M-worker count, 24 channel buffers. TT-Lang uses direct all-to-all PipeNets, single-block sequential staging and double-buffered compute operands. | Native mux buffering and TT-Lang DFB capacity describe different protocols; equal buffer counts would not imply equal buffering. Fabric transfers and compute blocks can be tuned independently. |
| Fabric initialization | Sweep uses STRICT_INIT and 8192-byte router payload; Blackhole operation test uses 4096-byte payload. | RELAXED_INIT, installed runtime's default router payload. | Router configuration and link-health requirements differ. |
| Timing | Compile/warmup, trace replay, `device_kernel_duration`, mean across devices. | Same device metric and mean across participants; ordinary launches, median across repeated samples. Per-sample maximum also retained. | No trace replay: TT-Lang replaces global-semaphore resources with synchronization. Cross-device dispatch skew can contribute device-side fabric waits. |
| Profiling implementation | `run_device_profiler`, Tracy signposts and ops-log processing. | `import_log_run_stats` with the standard `device_kernel_duration` analysis; one archived dump per sample. | Same per-device interval definition, different collection workflow; not whole-application latency. |
| Tensor residency and bias | TILE/interleaved DRAM; row bias. Test replicates each rank's KxN weight and 1xN bias. | TILE/interleaved DRAM; row bias; distinct N shards of weight/bias. | Same per-rank dimensions, different cross-rank values. TT-Lang additionally writes the local activation shard to gather scratch. |
| Communication/compute overlap | Consume available K blocks while communicating. | Gather one common fabric/compute K group during the first N round, then publish its operands to compute. Later N rounds reuse gathered DRAM data. | TT-Lang still stages local and remote activation through DRAM and uses separate send/receive staging. Native reads its local shard directly; the schedules and overlap are not identical. |
| Data-movement assignment | With transpose, activation uses BRISC/NoC 0; weight and output use NCRISC/NoC 1. | Activation and output use NCRISC/NoC 0; weight and fabric send use BRISC/NoC 1. | Operand NoCs match; processor assignment, fabric-send ownership and output-write ownership differ. Moving TT-Lang output writes to NoC 1 regressed N=1280 by 5.1% and was rejected. |

The sweep's M labels need care: its AGMM allocation multiplies M by the parent
sequence-parallel extent after selecting a singleton sequence-parallel submesh.
Use the explicit operation-test tensor dimensions above; do not infer actual
allocation dimensions from sweep test names alone.

## Run

Inside the fabric container, after verifying devices are unused:

```bash
source build-docker/env/activate
set -o pipefail
timeout 360 python -m benchmarks.all_gather_minimal_matmul \
    --dtype bf16 --samples 5 --json /tmp/all_gather_matmul_bf16.json \
    2>&1 | tee /tmp/device_test.log
```

Use `--dtype fp32`, `--implementation ttlang`, or
`--implementation ttmetal` as needed. The parent enables device profiling
before importing TTNN in each worker. Each implementation runs in a separate
process with a unique profiler directory beside the requested JSON.
Raw logs and individual worker reports are retained; existing logs are not deleted.

The default is global M=64/K=64/N=256, two devices, a 4x2 compute grid per
device, and bias enabled. Activations are K-sharded; weights, bias and output
are N-sharded. All tensors are TILE/interleaved DRAM. The driver discovers
the topology and selects a connected two-device submesh.

Tile/block parameters are exposed by `--help`.
`--k-tiles-per-device 4` exercises repeated transfers and a longer reduction.
Native non-transposed scheduling requires M <= per-device N and at least four
N workers; the driver rejects unsupported configurations.

The native plain-bias dimensions can be measured on the local pair with:

```bash
timeout 360 python -m benchmarks.all_gather_minimal_matmul \
    --m-tiles 96 --k-tiles-per-device 80 --n-tiles-per-device 40 \
  --m-block-tiles 2 --k-tiles-per-transfer 40 --k-block-tiles 40 \
  --n-block-tiles 2 \
    --worker-grid 2 5 --transpose \
    --json /tmp/all_gather_matmul_wan3072_n1280.json \
    2>&1 | tee /tmp/device_test.log
```

Use `--n-tiles-per-device 120` for per-device N=3840. These commands match
the native tensor dimensions, not its four-device ring or 12x9 worker grid.
The [results](PERFORMANCE.md#native-sized-local-comparison) retain those
distinctions alongside every reported ratio.

## Measurement contract

The analysis is the existing TT-Metal
[`device_kernel_duration`](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tools/tracy/device_post_proc_config.py)
used by its profiler reports. The processor is the same API used in
[TT-Metal's benchmarks](https://github.com/tenstorrent/tt-metal/blob/ea042c4ad6237678103cd7cbceb346e060f0f9a3/tests/ttnn/unit_tests/benchmarks/test_benchmark.py);
the selected metric includes all worker RISCs, not only TRISC1 math time.

- For each device, the metric spans the earliest kernel start to the latest
  kernel end across workers. Each sample reports the mean device duration,
  matching the native collective sweep. The maximum is retained as a diagnostic;
  `--device-aggregation max` reproduces the historical aggregation.
  Device clocks are never subtracted across chips.
- Three warmups precede five measured invocations by default. Each invocation
  is synchronized and its profiler buffers flushed before validation.
  The newest operation on each participating device is selected; launch IDs
  must advance between samples. Setup programs and warmups are excluded.
- Raw cycles, clock frequency from the profiler header, per-device launch IDs,
  derived microseconds, minimum/median/maximum, and correctness are retained.
  The comparison is the ratio of median durations, not host wall time.
- Compilation, Python dispatch, allocation, semaphore-initialization programs,
  host synchronization, validation, and profiler processing are outside the
  kernel interval. Worker waits within that interval remain included.
- This is not whole-application latency or trace replay. Current TT-Lang
  fabric resource replacement synchronizes between invocations, which prevents
  trace capture; neither implementation is measured through trace replay.
- Inputs, mesh, compute grid, M/K/N blocks, fidelity (HiFi2 for BF16, HiFi4
  for FP32), and FP32 destination precision match. `--no-fp32-dest-acc`
  explicitly selects BF16 accumulation for BF16
  tensors; it is not the native reference mode and fails numerical checks at
  large K. Tolerances are not relaxed for that mode.
- Native uses one linear fabric link, 24 channel buffers, one worker per M
  block, and additional fabric mux cores. TT-Lang uses direct fabric PipeNets
  on column-zero workers. Both use `FABRIC_1D`.
- TT-Lang writes the complete gathered activation to DRAM. Native scratch holds
  remote shards only; native compute reads its local shard directly. Validation
  checks these respective contracts. Kernel timings include TT-Lang's extra
  local DRAM traffic.
- Native uses packer L1 accumulation. TT-Lang materializes block products,
  converts them into the accumulation dtype, and sums through explicit DFBs.
  Bias conversion is separate from addition because the compiler rejects
  mixed-dtype fusion with an unrelated FP32 accumulator input.
  Both outputs must pass identical numerical criteria.

BF16 checks PCC >= 0.99 and rtol/atol 0.05; FP32 checks PCC >= 0.999 and
rtol/atol 0.005. FP32 FPU sources truncate to TF32. This does not claim
full-FP32 arithmetic accuracy. Gather scratch must match exactly.
A failed check exits nonzero without replacing the combined result report.

Reports record UTC time, source revision/status and hashes, binary hashes,
profiler source hashes, topology, device IDs, and compute settings.
Dependency pins do not prove the revision of installed binaries.
Set `BENCHMARK_CONTAINER_IMAGE` to the image digest for publication.

Historical HiFi4/two-buffer results use different settings. They must not be
compared directly with current defaults as a compiler performance regression.
Use `--math-fidelity HiFi4 --native-channel-buffers 2 --device-aggregation max`
and, for BF16, `--no-fp32-dest-acc`, with the same workload to reproduce the
historical settings. The operation implementation has changed; its historical
source revision is required to reproduce the original generated kernels.

## Host diagnosis

```bash
source build-docker/env/activate
set -o pipefail
timeout 180 python -m benchmarks.all_gather_minimal_matmul.profile \
    --implementation ttlang --runs 30 --output /tmp/ttlang.pstats \
    2>&1 | tee /tmp/device_test.log
```

Leave `TT_METAL_DEVICE_PROFILER` unset for host-only diagnosis.
