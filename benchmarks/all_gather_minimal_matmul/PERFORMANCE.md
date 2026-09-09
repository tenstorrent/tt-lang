# All-gather matmul performance diagnosis

Each results section records its own source, binary and timing provenance.
Dependency pins do not establish the build revision of the installed binaries.

## L1 activation reuse and FP32 packer accumulation

New raw reports and rejected-experiment snapshots are retained outside the
repository; their upload to the linked artifact archive is pending.

The current operation eliminates gathered-activation DRAM storage and retains
each M block's full K extent in per-worker L1 across N rounds. Matmul products
accumulate directly into FP32 L1 through the existing packer-accumulation DSL.
There is no intermediate BF16 product DFB, conversion DFB or SFPU accumulation
inside the reduction. The operation now takes activation, weight, bias and
output only; it does not return a gathered tensor.

The controlled comparison retains the previous transposed 2x5 worker grid,
M/K/N blocks 2/40/2, two Blackhole participants (IDs 1 and 2), BF16/HiFi2,
FP32 destinations and bias. Three warmups precede five samples using the
standard mean device-kernel duration across ranks, with ordinary launches.

| M / full K / per-device N | Previous TT-Lang ms | Current TT-Lang ms (range) | Original native ms (K=2) | Matched native ms (K=40, range) | Current / original native | Current / matched native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3072 / 5120 / 1280 | 4.437 | 3.439 (3.434-3.448) | 7.075 | 3.690 (3.687-3.699) | 0.486 | 0.932 |
| 3072 / 5120 / 3840 | 11.358 | 9.040 (8.959-9.170) | 20.131 | 10.396 (10.311-10.449) | 0.449 | 0.870 |

2026-09-09 23:00:22-23:01:27 UTC; TT-Lang `7ad660d45227` (operation/driver SHA-256 `0718680279f9`/`4351194bcd19`); Metal/LLVM pins `ea042c4ad623`/`37aca9d384347`; compiler/TTNN/Metal binary SHA-256 `a31540fb2fad`/`62edde2b1f61`/`65380f11dc15`; image `6eaf96b4b00d5`; [artifact archive (new reports pending)](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088). Previous/original columns retain the historical measurements documented below.

TT-Lang improves by 22.5% and 20.4% over the previous implementation and is
6.8% and 13.0% faster than the matched native configuration. This is not a
claim against native's best grid/block configuration or its Galaxy trace
benchmark. Numerical thresholds are unchanged. Maximum TT-Lang output errors
fall from 0.03892/0.03943 to 0.03282/0.03351; mean errors are approximately
0.003394/0.003395. Removing the gathered output also removes its separate
bit-exact check; both complete matmul-plus-bias outputs remain checked.

### Required compiler change

Packer accumulation already exists: `reserved_block += product` lowers to
packer L1 accumulation. The BF16-only variant failed the full-size accuracy
bound because each accumulated result rounded to BF16. Selecting an FP32
reserved block with `+= typecast(activation @ weight, fp32)` previously failed
the mixed-format compute-argument check.

[`computeDSTCapacity`](../../include/ttlang/Dialect/TTL/IR/TTLOpsUtils.h) now
accepts mixed-format inputs only when they are existing explicit conversions
or exclusively matmul source operands. Matmul sources unpack independently
from the result's pack format; FP32 output still selects FP32 DST capacity.
Unconverted SFPU inputs remain rejected. The generated BF16-to-FP32 typecast
has no SFPU instructions on Blackhole; the FP32 output DFB selects the pack
format. Device tests exercise BF16/BF16, BF16/FP32 and FP32/FP32 packing with
single-tile, DST-sized and subblocked outputs. No LLVM or TT-Metal change is
required, and no new compiler option is added.

### Attribution and rejected changes

The updated implementation also benefits from a wider grid. At
M3072/K5120/per-device N1280, transposed 2x8 workers with M/K/N blocks 2/40/1
measure 2.793 ms (2.781-2.809), versus native's 3.627 ms (3.625-3.634) with
identical settings: ratio 0.770. This is 18.8% faster than the current 2x5
TT-Lang configuration, but changes both worker count and N blocking. Native's
earlier 2x8, 4/20/1 trial is faster at 3.283 ms; matching TT-Lang's blocks does
not identify native's best blocks. The 2x8 N3840 case is not measured.

2026-09-09 23:01:34-23:02:00 UTC; same committed sources, compiler/runtime binaries and timing contract as the table above; report `l1_grid2x8_n1280.json` retained outside the repository pending archive upload.

At M3072/K5120/per-device N1280 on the 2x5 grid, disabling activation reuse
increases TT-Lang from 3.439 to 5.347 ms (+55.5%). Both variants avoid DRAM
gather storage and use FP32 packer accumulation; the streamed variant repeats
activation reads, fabric transfers and row multicasts for each N round. It is
retained as an explicit lower-L1-capacity option, not selected for these results.

Repeated matmul initialization is not a material bottleneck in this workload.
A compiler experiment removed the per-K initialization only after proving
invariant operands/dimensions and preserved unpack/math configuration. An
alternating 20-pair generated-C++ comparison at K=10 measured repeated/hoisted
medians of 4.684/4.680 ms and a median paired hoisted/repeated ratio of 0.9993.
The only C++ difference restored the identical initialization call inside K.
The experiment uses K=10 because keeping both variants' fabric resources
resident at K=40 exceeded the second compilation's remaining L1 budget.
A separate K=20 comparison measured 3.797/3.817 ms, a 0.5% regression with
overlapping ranges. The initialization transformation is not retained; its
source, tests and measurements are retained locally pending archive upload. The final
numbers above use the retained compiler, with per-K initialization unchanged.

The complete fabric correctness matrix covers cached/streamed activation,
BF16/FP32, bias/no bias, transpose, repeated output blocks and several K block
sizes on the two-device submesh. Larger meshes are not device-tested here.
The small-workload comparison has not been rerun for this implementation.
Further device work stopped when another benchmark acquired the devices;
the attempted 2x10 run failed during UMD initialization before compiling or
executing the operation. It is not evidence of a grid-specific failure.

## Operation-only optimization

The original gap has substantial implementation-level causes. No compiler,
runtime, toolchain or timing-method change is needed for the improvements below.
Compute K blocking, fabric transfer granularity and data-movement assignment
are varied separately before selecting the combined configuration.

Both implementations use BF16/HiFi2, FP32 destinations, M/K/N blocks 2/40/2,
and the same transposed 2x5 worker grid on two Blackhole devices. TT-Lang fabric
transfers contain 40 K tiles. Other settings retain the
[measurement contract](README.md#measurement-contract): three warmups, five
samples, mean device-kernel duration across ranks, ordinary launches.

| M / full K / per-device N | Original TT-Lang ms | Optimized TT-Lang ms (range) | Original native ms (K=2) | Native ms (K=40, range) | Optimized TT-Lang / original native | Optimized TT-Lang / K=40 native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3072 / 5120 / 1280 | 24.456 | 4.437 (4.431-4.439) | 7.075 | 3.696 (3.691-3.700) | 0.627 | 1.201 |
| 3072 / 5120 / 3840 | 63.013 | 11.358 (11.355-11.362) | 20.131 | 10.385 (10.360-10.427) | 0.564 | 1.094 |

2026-09-09 21:40:10-21:41:16 UTC; TT-Lang `0b4b8840c431` + local changes (operation SHA-256 `730491f4be48`, driver `4b712b8cde1f`); Metal/LLVM pins `ea042c4ad623`/`37aca9d384347`; compiler/TTNN/Metal binary SHA-256 `7f5e02a65e4c`/`62edde2b1f61`/`65380f11dc15`; image `6eaf96b4b00d5`; [reports and exact sources](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088).

The K=40 native measurements use the same TTNN operation and 2x5 grid as the
original native baseline, with only compute K blocking increased from 2 to 40
tiles. No native implementation code changes. Optimized TT-Lang is 37.3% and
43.6% faster than the original K=2 native configuration; those ratios compare
different blocking. The K=40 native column compares matching compute blocks.

The original native configuration is the locally constrained baseline, not
upstream's tuned configuration. TT-Lang experiments motivated changing K
blocking for both implementations; they did not introduce a native algorithm
or demonstrate an optimization absent upstream. TT-Metal already provides a
[block-size sweep](https://github.com/tenstorrent/tt-metal/blob/9fe0ba04fc9d450555b756f35dcb4de88126693d/models/tt_dit/utils/sweep_mm_block_sizes.py),
and its [plain-bias tests](https://github.com/tenstorrent/tt-metal/blob/9fe0ba04fc9d450555b756f35dcb4de88126693d/models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py#L637-L642)
use 8/8/8 blocks for these dimensions.

Source check on 2026-09-09: fetched TT-Metal `origin/main` at
`9fe0ba04fc9d450555b756f35dcb4de88126693d` and compared all 13 files in the
[native operation directory](https://github.com/tenstorrent/tt-metal/tree/9fe0ba04fc9d450555b756f35dcb4de88126693d/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async)
with the installed v1.1.9 sources. The only differences are five
`copy_tile_to_dst_init_short` -> `copy_init` substitutions in `compute.cpp`;
the program factory and data-movement kernels are byte-identical. The sweep
and operation test are unchanged from the earlier `f69f924c6b4f` reference.
This comparison does not cover dependencies outside the operation directory
or establish binary equivalence. Current-main binaries have not been measured.

TT-Lang improves by 5.51x and 5.55x, but remains 20.1% and 9.4% slower than
native with the same K=40 blocks. Accuracy thresholds are unchanged and the
gather remains bit-exact. Mean absolute output errors are 0.003634/0.003636;
maximum errors increase from 0.03290/0.03636 to 0.03892/0.03943, still within
the original bounds. Native's 12x9 Galaxy/trace configuration is not measured
on this machine.

A separate alternating 20-pair check of the original and optimized operations
on the non-transposed M64/K64/per-device N128 smoke case measured medians
13.970/14.084 us (0.8% slower), with ranges 12.530-15.281/13.068-15.073 us.
The median paired ratio was 0.982. This does not establish a consistent small-
workload speedup or regression; both statistics are retained in the archive.
That check uses the same BF16/HiFi2/FP32-destination settings and device metric.

### Attribution experiments

M=3072, full K=5120, per-device N=1280; the 2x5 transposed grid and 2x2 output
blocks are unchanged except in the explicitly labeled wider-grid experiment.
Every reported time passes the original correctness checks. Times are median
device-kernel milliseconds, not host latency.

| Implementation/configuration change | TT-Lang ms | Native ms | Interpretation |
| --- | ---: | ---: | --- |
| Original compute K=2, fabric K=2 | 24.456 | 7.075 | Baseline. |
| Compute K=20, fabric K=2 | 9.478 | 3.714 | Fewer compute invocations, intermediate packs, local multicasts and DRAM barriers; fabric message count is unchanged. |
| Compute K=40, fabric K=2 | 9.357 | 3.694 | Diminishing benefit from compute blocking alone. |
| Compute K=40, fabric K=20 | 5.337 | unchanged | Larger fabric transfers and local staging batches remove approximately 4.02 ms; this does not isolate wire-transfer time from staging and synchronization. |
| Fabric K=40, single-block sequential staging | 5.111 | unchanged | Lower staging capacity permits larger transfers. |
| Publish each gathered K group to compute | 5.083 | 3.696 | Small benefit at this blocking; does not eliminate DRAM staging. |
| Activation NoC 0, weight NoC 1 | 4.437 | unchanged | Data-movement assignment matters; this also swaps RISCs and fabric-send ownership, not just NoC direction. |
| Move output writes to the weight thread | 4.661 | 3.696 | **5.1% regression**; rejected. |
| Separate trial: 2x8 grid, M/K/N=4/20/1, fabric K=20 | 5.323 | 3.283 | TT-Lang is slower despite 16 rather than 10 workers; rejected as the local TT-Lang configuration. |

2026-09-09 20:54-21:33 UTC; base `0b4b8840c431` plus operation edits (original baseline `e0cced786d1e` plus edits); source hashes, exact configurations and binary identities are recorded in each [archived report](https://gist.github.com/brnorris03/79c57b196efe09355699d40165780088).

The native column uses each changed compute block. `unchanged` means
the preceding native compute configuration is unchanged, not that a new native
sample was collected. The native 16-worker result is faster than its 10-worker
result: matching TT-Lang's best grid does not establish native's optimal grid.
These are successive controlled changes, not additive measurements of disjoint
hardware components; communication and compute can overlap.

### Generated compute and remaining opportunities

For each 2x2 output block, the original operation performs 80 compute K
iterations. Each iteration packs four BF16 product tiles, converts and packs
four FP32 partial tiles, then reloads/adds/packs four FP32 accumulator tiles:
960 tile packs per output block, excluding initialization and bias/output.
Native packs four FP32 intermediate tiles per iteration with L1 accumulation:
320 tile packs. At compute K=40, these counts become 48 and 16 respectively;
the number of matrix tile products is unchanged. Larger blocks also reduce
operand DFB handoffs and per-block initialization. This explains why changing
K blocking helps both implementations, especially TT-Lang.

The archive includes `compute_baseline_n1280.cpp`,
`compute_streamed_n1280.cpp` and `native_compute_installed.cpp` for these counts;
the historical smoke snapshots below describe a different implementation.

That earlier operation rounded each matmul-block product to BF16 before
widening to FP32, retained extra conversions and SFPU additions, and staged
gathered activation through DRAM. The current implementation above removes
those costs. Larger or more deeply buffered blocks must fit the actual L1
budget; adding workers alone did not improve that earlier implementation.

At that earlier revision, existing DSL alternatives were tested before
proposing compiler work:

| Attempt | Observed limitation | Work needed to enable that specific optimization |
| --- | --- | --- |
| BF16 tensor recurrence over K | Compiles to BF16 packer accumulation, but the full-size N=1280 output fails the unchanged tolerance at 9 elements; max absolute error 0.151. | An FP32 intermediate or proven DST-resident matmul recurrence is needed; relaxing accuracy is not an optimization. |
| FP32 reserved accumulator `+= typecast(activation @ weight, fp32)` | [`computeDSTCapacity`](../../include/ttlang/Dialect/TTL/IR/TTLOpsUtils.h) rejects the mixed BF16/FP32 compute arguments. | Support BF16 matmul inputs with an FP32 packed result, including correct DST capacity, unpack/pack formats and conversion semantics. Removing the diagnostic alone is insufficient. |
| Materialize the product, then accumulate its conversion in the same loop | [`collectDFBAccumulationStores`](../../lib/Dialect/TTL/Transforms/TTLInsertAccumulationScopes.cpp) rejects a non-accumulating store inside a packer-accumulation loop (#648). | Model packer accumulation state per store so ordinary product packs cannot accidentally accumulate. |
| Hold a waited accumulator across K and update it in place | [`ComputeOpCreationPlanning.cpp`](../../lib/Dialect/TTL/Transforms/ComputeOpCreationPlanning.cpp) requires waited replacement to execute in a straight-line entry block. | Prove loop-scoped consumer ownership and repeated replacement before permitting this form. |

No rejected variant was retained at that revision. The current implementation
resolves the FP32 packed-result restriction as described above; the other
limitations are not prerequisites for this optimization. Native's FP32
intermediate format is confirmed in the installed program factory, not inferred
from the destination precision flag alone.

## Original native-sized local comparison

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

## Historical FP32 numerical checks

FP32 results also pass the documented FPU-aware checks: TT-Lang 3.440 ms/call,
native 0.106 ms/call. Both failed the initial rtol/atol 0.0001 comparison against
full-FP32 PyTorch: maximum absolute errors were 0.005454 and 0.007277 respectively.
FP32 FPU sources truncate to TF32; TT-Lang additionally reloads partial sums
through the source registers. The benchmark uses rtol/atol 0.005 and PCC >=
0.999 for FP32, versus 0.05 and 0.99 for BF16, and records absolute errors.
This does not claim full-FP32 arithmetic accuracy.

## Remaining host/runtime work

Replay-safe resource reuse and immutable binding-plan caching remain necessary
to address the historical host overhead and support trace capture. Neither is
implemented here, and host timings are not used to explain the measured device
speedups. Device-level implementation changes and the remaining compiler
constraints are described in the operation-only analysis above.
