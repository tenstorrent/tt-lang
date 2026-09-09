# All-gather matmul performance diagnosis

Each results section records its own source, binary and timing provenance.
Dependency pins do not establish the build revision of the installed binaries.

## Native-sized local comparison

The [configuration table](README.md#comparison-with-the-native-benchmark)
distinguishes matching settings from the native Galaxy benchmark. The following
measurements use the native test's plain-bias tensor dimensions, with BF16
inputs/output, HiFi2 and FP32 destinations. Both implementations use two
Blackhole devices (IDs 1 and 2), a transposed 2x5 worker grid, 2/2/2 M/K/N tile
blocks, TILE/interleaved DRAM, and ordinary launches. Native uses one link,
two workers/link, 24 channel buffers, and 2x2 subblocks. Five samples follow
three warmups; each sample is the mean device-kernel duration across ranks.

| M / full K / per-device N | TT-Lang median ms (range) | Native median ms (range) | TT-Lang/native |
| --- | ---: | ---: | ---: |
| 3072 / 5120 / 1280 | 24.456 (24.450-24.457) | 7.075 (7.071-7.092) | 3.457 |
| 3072 / 5120 / 3840 | 63.013 (63.010-63.017) | 20.131 (20.122-20.152) | 3.130 |

2026-09-09 20:54:51-20:57:34 UTC; TT-Lang `e0cced786d1e` + local changes (operation SHA-256 `a27b232a779b`, driver `376ade3e8dbd`); Metal/LLVM pins `ea042c4ad623`/`37aca9d384347`; compiler/TTNN/Metal binary SHA-256 `7f5e02a65e4c`/`62edde2b1f61`/`65380f11dc15`; image `6eaf96b4b00d5`; [reports and measured sources](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088).

TT-Lang is 245.7% and 213.0% slower than this locally constrained native
configuration. These are not results against native's tuned 12x9 grid or trace
replay, and are not comparable to the tiny historical cases below as a
same-workload regression. Native performance is also limited by the shared
local grid and block settings.

All output and gather checks pass without relaxing tolerances. TT-Lang's mean
absolute output error is 0.00365 for both cases. The earlier BF16-accumulator
implementation failed N=1280 with mean absolute error 0.0679 and 2,282,733 of
7,864,320 elements outside the existing bound. Explicit FP32 accumulator/bias
DFBs correct that failure; matmul-block products still round to the input dtype
before FP32 addition, unlike native's packer accumulation.

Unsuccessful larger configurations remain part of the evidence: 8/8/8 blocks
exceeded L1; twelve M workers exceeded available forwarding connections;
4/8/8 blocks on the 2x5 grid timed out. The timed-out processes exited before
the two participant boards were reset, after verifying all devices were idle.
No timing result is reported for those configurations. The exact cause of the
4/8/8 stall remains unresolved.

## Historical smoke measurements

M=64/K=64/N=256, two Blackhole devices (IDs 1 and 2), 4x2 compute workers
per device, 1x1x1 tile blocks, HiFi4, bias enabled. The
discovered topology is 2x2; the operation uses its 2x1 submesh. All numerical
checks pass. See the [measurement contract](README.md#measurement-contract).

The comparison now uses TT-Metal's existing `import_log_run_stats()` processor
and `device_kernel_duration` analysis, not a local CSV parser or host timer.
Five individually profiled invocations follow three warmups. Each sample is
the maximum duration across participating devices; the table reports medians.
Clock conversion uses the profiler header's 1350 MHz.

| Dtype | TT-Lang device us (range) | Native device us (range) | TT-Lang/native |
| --- | ---: | ---: | ---: |
| BF16 | 15.994 (14.276-19.395) | 14.558 (12.678-16.443) | 1.099 |
| FP32 | 17.321 (16.874-17.897) | 16.265 (12.873-16.547) | 1.065 |

2026-09-09 18:42:26-18:44:31 UTC; TT-Lang `3e688e1f07b7` + local example/benchmark; Metal pin `ea042c4ad623`, LLVM pin `37aca9d384347`; compiler/TTNN/Metal binary SHA-256 `7f5e02a65e4c`/`62edde2b1f61`/`65380f11dc15`; image `6eaf96b4b00d5` ([BF16 report](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/ca8ce847d3b44433a3a69fec4738f33067ba9be7/all_gather_bf16_device_final.json), [FP32 report](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/6d77f0465c570640a93615f47e1c2cd5dcb277ac/all_gather_fp32_device_final.json)).

The latest medians are 9.9% slower for BF16 and 6.5% slower for FP32. Earlier
standard-processor runs measured ratios of
[0.966 for BF16](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/6ce6c35711f5b10b6071ed3383dab135810e9187/all_gather_bf16_device_standard.json)
and [1.305 for FP32](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/a466115e562e50305a96f312d052957d2ce5b61b/all_gather_fp32_device_standard.json).
The final driver archives and processes each dump separately instead of
reprocessing the growing log; all profiler processing remains outside timing.
Sample variability prevents attributing these changes to that host-side edit.
Neither run supports a 27-33x kernel slowdown. These measurements are not
evidence of an improvement against historical host timings. Kernel and compiler
code are unchanged between the earlier diagnostic and these measurements.

## Historical host measurements

The following data is retained for host/runtime diagnosis only. The original
comparison emphasized host call time; it did not establish kernel performance.

| Measurement | TT-Lang | Native | TT-Lang/native |
| --- | ---: | ---: | ---: |
| Initial uninstrumented median call | 3476.6 us | 104.4 us | 33.3 |
| Repeat uninstrumented median call | 3551.0 us | 129.4 us | 27.4 |

End-to-end samples each contain 10 calls after 3 warmups; the table reports
the median of 5 samples. Native dispatch is short enough that host variability
materially changes the ratio. The repeated TT-Lang result remains about 3.5 ms.
The previous custom-parser device result (13.716 vs 10.927 us, 1.26x) is
superseded by the standard-processor measurements above. Its raw logs remain
archived. It used different sampling/flush behavior and is not a regression
baseline for the new run.

## Host/runtime attribution

A separate cProfile run measured 30 warmed TT-Lang calls at 4.925 ms/call.
Profiling adds Python overhead, so these numbers explain attribution rather
than replacing uninstrumented timings. The following cumulative groups are
disjoint:

| Runtime work | Profiled time per call |
| --- | ---: |
| Fabric target binding plans, two devices | 1.532 ms |
| Runtime resource cache lookup and replacement | 1.196 ms |
| DFB descriptor construction and validation | 0.923 ms |

These groups account for 74% of the profiled call. Resource replacement includes
0.628 ms in global semaphore construction and 0.224 ms in synchronized release;
these are subsets of the 1.196 ms, not additional costs.

Source evidence:

- [`_get_cached_runtime_resources_impl`](../../python/ttl/kernel_runner.py)
  reuses compatible resources only when the global-semaphore count is zero.
  Fabric calls release and reconstruct them even for identical input tensors.
  `_release_cached_runtime_resources_impl` synchronizes before releasing owners.
- `_run_kernel_on_device_impl` reconstructs DFB/kernel/program descriptors and
  calls `build_fabric_target_binding_plan` for each participating device on
  every invocation. Route lookup caching does not cache the complete binding
  plan or descriptor construction.
- The native benchmark constructs its global semaphores once, outside timing,
  and repeatedly invokes the native cached operation.

The synchronization cannot simply be removed: it protects in-flight resource
ownership. It also makes the current TT-Lang fabric invocation incompatible
with trace capture. Correct resource reuse needs a replay-safe initialization
protocol and stable ownership, not a benchmark-only bypass.

## Historical generated C++ versus native kernels

The source snapshots used for comparison are retained in
[the measurement archive](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088).
They describe the earlier smoke implementation, before fixed-grid output loops
and explicit FP32 accumulator/bias DFBs. Exact pack counts below do not describe
the native-sized implementation.

| Mechanism | TT-Lang generated C++ | Native C++ |
| --- | --- | --- |
| Activation availability | Write local and remote shards to DRAM; finish gather; read DRAM again into the compute DFB. | Read the local shard directly and stream available K blocks into compute while forwarding fabric data. |
| Accumulator storage | Copy between two intermediate DFBs around each K iteration; reload the accumulator into DST. | Reserve one intermediate block; use packer L1 accumulation across K blocks. |
| Compute initialization | Full matmul startup inside each K iteration; repeated format changes around copies. | Startup once, matmul block initialization outside the K loop. |
| Bias/output | Broadcast plus SFPU add into an intermediate DFB, then copy to output. | Fused row-broadcast add packs directly into the output DFB. |

For the default two-K-block case, TT-Lang executes nine pack operations per
worker: initial zero fill, three per K block, bias result, and final output
copy. Native executes three: one per K block and the final bias/output pack.
See [TT-Lang compute](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/907154a4e34740fb51a38885b66db4a3b6df39d1/ttlang_compute_bf16.cpp)
(lines 34-117), [native compute](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/8a019b2751c9c9a6249eec1fad13939a3478c326/native_compute.cpp)
(`matmul_blocks`, `kernel_main`, `add_bias_block`), and
[native activation](https://gist.githubusercontent.com/brnorris03/79c57b196efe09355699d40165780088/raw/f064d1b1f316e757321f31efdde42e75db1a66a3/native_activation.cpp)
(lines 425-455, including publication to compute before forwarding).

These differences explain additional work and reduced overlap, but the
measured device gap is only a few microseconds for this case. No experiment
isolated the latency contribution of each individual C++ difference.

## Numerical precision

FP32 results also pass the documented FPU-aware checks: TT-Lang 3.440 ms/call,
native 0.106 ms/call. Both failed the initial rtol/atol 0.0001 comparison against
full-FP32 PyTorch: maximum absolute errors were 0.005454 and 0.007277 respectively.
FP32 FPU sources truncate to TF32; TT-Lang additionally reloads partial sums
through the source registers. The benchmark uses rtol/atol 0.005 and PCC >=
0.999 for FP32, versus 0.05 and 0.99 for BF16, and records absolute errors.
This does not claim full-FP32 arithmetic accuracy.

## Optimization order

1. Retain invocation-compatible global semaphore and hidden DFB resources with
   a correct reset/ownership protocol. Cache immutable binding plans and
   descriptors; update only invocation-dependent addresses and arguments.
2. Remove the accumulator DFB copies and repeated compute initialization
   without changing required dtype/rounding semantics.
3. Stream activation blocks into matmul instead of completing a DRAM-staged
   gather first; retain full gathered output only when the operation contract
   requires it.

These are diagnosed optimization targets, not implemented runtime/compiler
changes. Larger workloads are required to quantify throughput effects.
