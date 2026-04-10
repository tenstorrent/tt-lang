# tt-lang Matmul Performance Study

A performance comparison of tt-lang compiled matmul kernels against
ttnn.matmul (tt-metal's hand-optimized implementation) on Tenstorrent
Blackhole hardware.

## 1. Experimental Setup

**Hardware.** Single Blackhole chip, 140 compute cores (130 usable with
storage grid). 1.35 GHz core clock. Per-core L1: 1.5 MB.

**Software.** tt-lang compiler (commit 2544cb1d), tt-metal toolchain
v0.1.8. All kernels compiled with `maximize_dst=true`,
`fp32_dest_acc_en=true`, `use_block_matmul=true`, bf16 data type.

**Benchmark script.** `examples/matmul_bench/bench_matmul.py`. Raw
data appended to `examples/matmul_bench/matmul_perf_results.csv` with
git SHA and timestamp for reproducibility.

**Correctness.** Every benchmark configuration is validated against an
f32 golden reference (`a.float() @ b.float()`) with PCC > 0.99. The
benchmark aborts if any configuration fails the correctness check.

### 1.1 Measurement Protocol

Each configuration is measured as follows:

1. **Warmup:** 3 iterations (triggers JIT compilation, populates
   caches).
2. **Timed runs:** 10 iterations. Each iteration is bracketed by
   `ttnn.synchronize_device()` calls so that the measured interval
   captures full device execution, not host dispatch latency alone.
3. **Reported statistics:** median, min, max over the 10 timed runs.
   The ratio column normalizes against the ttnn.matmul median for the
   same problem size.

### 1.2 Kernel Versions

Three tt-lang kernel versions are compared against ttnn.matmul. All
implement C = A * B (general matrix multiply) with identical numerical
results (PCC > 0.99 against f32 golden).

**ttnn.matmul** -- tt-metal's production matmul. Uses L1 accumulation
(`packer_l1_acc=True`), subblocking (subblock_h=2, subblock_w=2),
and K-direction alternation for input reuse. The internal DMA scheduling
strategy is not directly observable from the Python API; the description
in Section 6 is derived from reading the tt-metal source code.

**v1-single-reader** -- tt-lang kernel with all DMA reads on one RISC
core. The reader thread fetches both A and B blocks; the writer thread
only writes output. Compute is identical to v2-split-dma: user-managed
outer K loop with `acc.store(prev + a @ b)`.

**v2-split-dma** -- Same compute kernel as v1, but DMA is split across
both RISC cores: NCRISC reads A blocks, BRISC reads B blocks and writes
output. Block sizes are parameterized by M_block, K_block, N_block.
When the output block exceeds DST capacity, `SubblockComputeForDST`
partitions the M x N output block into DST-sized subblocks. K
accumulation remains in DST within each subblock (matmul_block
accumulates kt tiles in-place).

**v3-compiler-k** -- tt-lang kernel where the full K dimension is
placed in one DFB fill. The user writes `out.store(a @ b)` and the
compiler generates the K reduction loop internally from the 3D
[M, N, K] iteration space. No user-managed K loop or accumulator DFB.
Only viable when M_block * K + K * N_block fits in L1.

### 1.3 Isolating Compute from Data Movement

Two memory configurations are used to separate compute and DMA
contributions:

- **DRAM:** Tensors in interleaved DRAM. Measured time includes DMA
  transfers between DRAM and L1 circular buffers.
- **L1:** Tensors in interleaved L1 (`ttnn.L1_MEMORY_CONFIG`). DMA
  reads from L1 instead of DRAM. The remaining time reflects compute
  kernel execution, CB synchronization overhead, and L1 NOC transfer
  latency. Note: L1 interleaved still involves NOC transfers between
  cores, so this does not fully eliminate data movement; it eliminates
  DRAM bandwidth as a variable.

## 2. Compute Kernel Structure

This section describes the generated compute kernel for each version.
The pseudocode reflects the EmitC output after all compiler passes.
Data movement kernels are identical across compute versions and omitted
here; their impact is analyzed in Section 4.

All versions share the same outer loop structure over output blocks:

```
for each (m_block, n_block) assigned to this core:
    for kb in 0..K_num_blocks:
        cb_wait(in0);  cb_wait(in1)      // wait for DMA to fill inputs
        <compute body>                    // version-specific
        cb_pop(in0);   cb_pop(in1)
    cb_push(out)                          // signal output ready for DMA
```

The versions differ in how `<compute body>` handles DST register
management and K accumulation.

### 2.1 ttnn.matmul (reference)

ttnn.matmul's compute kernel (`minimal_matmul/device/kernels/compute.cpp`)
uses the following structure per output block (derived from reading the
tt-metal source code, not from runtime observation):

```
cb_reserve_back(intermediate_cb, M_block * N_block)
for kb in 0..K_num_blocks:
    cb_wait(in0); cb_wait(in1)
    for ms in 0..M_block step subblock_h:         // subblock iteration
        for ns in 0..N_block step subblock_w:
            tile_regs_acquire()
            for k in 0..K_block:                   // DST accumulation
                matmul_block(in0, in1, ms, ns, dst=0,
                             ct=subblock_w, rt=subblock_h, kt=K_block)
            tile_regs_commit(); tile_regs_wait()
            for h in 0..subblock_h:                // indexed pack
                for w in 0..subblock_w:
                    pack_tile<true>(h*subblock_w + w, intermediate_cb,
                                   (ms+h)*N_block + (ns+w))
            tile_regs_release()
    cb_pop(in0); cb_pop(in1)
    if kb == 0: pack_reconfig_l1_acc(1)            // L1 accumulation
cb_push_back(intermediate_cb, M_block * N_block)
pack_reconfig_l1_acc(0)
```

DST holds `subblock_h * subblock_w` tiles (e.g., 2x2=4 with
`fp32_dest_acc_en=true`, or 8 without).
Within each subblock, `matmul_block` accumulates K_block tiles via DST.
Across K_num_blocks iterations, `pack_reconfig_l1_acc` makes each pack
additive to the existing L1 value, avoiding the need to reload the
accumulator from L1 into DST between K blocks.

### 2.2 v2-split-dma (current tt-lang, K_block > 1)

The `prev + a @ b` user pattern lowers to `copy_tile` (load accumulator
from CB to DST) followed by `matmul_block` (DST += A*B). With
subblocking enabled (commit 2544cb1d), the compiler generates the
following structure (verified by inspecting the EmitC output):

```
// First K block: standalone matmul
for ms in 0..M_block step subblock_h:
    for ns in 0..N_block step subblock_w:
        tile_regs_acquire()
        matmul_block(in0, in1, ms, ns, dst=0,
                     ct=subblock_w, rt=subblock_h, kt=K_block)
        tile_regs_commit(); tile_regs_wait()
        for h, w in subblock:
            pack_tile<true>(dst[h*sw+w], acc_cb, indexed)
        tile_regs_release()

// Remaining K blocks: copy_tile + matmul_block
for kb in 1..K_num_blocks:
    for ms in 0..M_block step subblock_h:
        for ns in 0..N_block step subblock_w:
            tile_regs_acquire()
            for h, w in subblock:
                copy_tile(acc_cb, h*sw+w, dst[h*sw+w])  // reload accumulator
            matmul_block(in0, in1, ms, ns, dst=0,
                         ct=subblock_w, rt=subblock_h, kt=K_block)
            tile_regs_commit(); tile_regs_wait()
            for h, w in subblock:
                pack_tile<true>(dst[h*sw+w], acc_cb, indexed)
            tile_regs_release()

// Copy acc to output
for ms, ns subblocks:
    tile_regs_acquire()
    for h, w in subblock:
        copy_tile(acc_cb, h*sw+w, dst[h*sw+w])
    tile_regs_commit(); tile_regs_wait()
    for h, w in subblock:
        pack_tile<true>(dst[h*sw+w], out_cb, indexed)
    tile_regs_release()
```

Structural differences from ttnn.matmul:

- **copy_tile per K iteration.** Each K block after the first reloads
  the accumulator from the acc CB into DST via `copy_tile`. ttnn.matmul
  avoids this with L1 accumulation (`pack_reconfig_l1_acc`).
- **Extra acc CB.** The intermediate accumulator requires a separate CB
  (`acc_dfb`). ttnn.matmul writes directly to `intermediate_cb` with
  L1 additive packing.
- **Final copy.** The accumulated result must be copied from `acc_cb` to
  `out_cb`. ttnn.matmul's post-op operates on `intermediate_cb` directly.

### 2.3 v2-split-dma (K_block = 1)

With K_block=1, each outer K iteration contains a single matmul_block
call with kt=1. The structure is identical to Section 2.2 but with
K_block=1, meaning:

- `matmul_block` executes one K tile per call (minimal work per call)
- K_num_blocks = Kt (one outer iteration per K tile)
- More `copy_tile` calls (one per K tile per subblock)
- More `tile_regs_acquire`/`release` cycles

### 2.4 v3-compiler-k (L1 accumulation, output > DST)

When the user writes `out.store(a @ b)` with full K in the DFB and the
output block exceeds DST capacity, the compiler generates a K reduction
loop with L1 accumulation. `LowerMatmulBlock` detects the subblocked
accumulating matmul (output M*N > DST capacity, K > 1, no explicit
accumulator) and emits the following structure (verified by inspecting
the generated TTKernel IR):

```
scf.for %k = 0 to K step 1 {                  // reduction loop
    lhs_slice = extract_slice(lhs, [0, %k], [M, 1])
    rhs_slice = extract_slice(rhs, [%k, 0], [1, N])
    dst_section {
        matmul_block(lhs_slice, rhs_slice, dst=0,
                     ct=N, rt=M, kt=1)         // single K tile
        for m in 0..M:
            for n in 0..N:
                pack_tile<true>(m*N+n, out_cb, indexed)
    }
} {ttl.reduction_loop}
```

`TTKernelInsertL1Accumulation` detects the `ttl.reduction_loop`
annotation and inserts guards:
- `pack_reconfig_l1_acc(1)` after `tile_regs_acquire` when the loop
  induction variable is not equal to the lower bound (i.e., from the
  second K iteration onward)
- `pack_reconfig_l1_acc(0)` after the loop exits

Differences from v2-split-dma (Section 2.2):

- **No copy_tile.** L1 additive packing replaces the explicit
  accumulator reload. Each K iteration packs to the same CB slot;
  `pack_reconfig_l1_acc` makes subsequent packs additive.
- **No acc CB.** The output is written directly to `out_cb`. No
  intermediate `acc_dfb` or final copy step.
- **Per-K DstSection.** DST is acquired and released each K iteration
  (the DstSection is inside the K loop). In v2-split-dma, DST persists
  across K iterations within the fused `copy_tile + matmul_block`
  DstSection.

This structure matches the tt-metal minimal_matmul pattern (Section
2.1) at the per-K-iteration level. The remaining structural difference
is that minimal_matmul's outer K loop is in the data movement layer
(cb_wait/cb_pop per K block), while v3-compiler-k's K loop is
compiler-generated within the compute kernel from the 3D iteration
space.

## 3. Results

### 3.1 L1-Only (Compute Isolation)

**Problem:** 2048 x 2048 x 2048 (64 x 64 x 64 tiles).
M_block = N_block = 8. Grid: auto (130 cores).
All tensors in interleaved L1.

| Version | K_block | K_num_blocks | Median (ms) | Min (ms) | Max (ms) | vs ttnn |
|---------|---------|--------------|-------------|----------|----------|---------|
| ttnn.matmul | -- | -- | 0.55 | 0.54 | 0.55 | 1.00x |
| v2-split-dma | 8 | 8 | 0.51 | 0.42 | 0.55 | 0.93x |
| v2-split-dma | 1 | 64 | 0.47 | 0.47 | 0.48 | 0.87x |

**Observation 1.** tt-lang v2-split-dma executes 7--13% faster than
ttnn.matmul in the L1 configuration (0.93x and 0.87x respectively).
This measurement eliminates DRAM bandwidth as a variable but does not
fully eliminate data movement: L1 interleaved tensors still require
NOC transfers between cores.

**Observation 2.** K_block=1 is 8% faster than K_block=8 in L1
(0.47 vs 0.51 ms). One possible explanation: each matmul_block call
with kt=1 has lower per-call latency than kt=8, and the additional
`copy_tile` and loop overhead per iteration does not offset this
reduction. However, without per-RISC-thread cycle counts (the device
profiler was not producing per-run data in this build), we cannot
confirm whether the difference is in the MATH thread (matmul_block
latency), the PACK thread (pack_tile latency), or the UNPACK thread
(copy_tile latency).

### 3.2 DRAM: End-to-End Performance

**Problem:** 4096 x 4096 x 4096 (128 x 128 x 128 tiles).
M_block = N_block = 8. Grid: auto (130 cores).
Matches tt-metal `test_minimal_matmul.py::test_linear` reference shape.

| Version | K_block | K_num_blocks | Median (ms) | Min (ms) | Max (ms) | vs ttnn |
|---------|---------|--------------|-------------|----------|----------|---------|
| ttnn.matmul | -- | -- | 0.90 | 0.75 | 1.03 | 1.00x |
| v1-single-reader | 8 | 16 | -- | -- | -- | -- |
| v1-single-reader | 1 | 128 | -- | -- | -- | -- |
| v2-split-dma | 8 | 16 | 3.44 | 3.34 | 3.51 | 3.82x |
| v2-split-dma | 1 | 128 | 3.19 | 3.10 | 3.26 | 3.54x |

v1-single-reader has not yet been re-measured with the current
benchmark script. The rows above will be populated once the
single-reader variant is added to `bench_matmul.py`.

**Observation 3.** tt-lang v2-split-dma is 3.5--3.8x slower than
ttnn.matmul in the DRAM configuration. The L1-only results (Section
3.1) show that the tt-lang compute kernel is not slower than ttnn's in
isolation. This is consistent with the hypothesis that the DRAM
slowdown originates in the data movement layer, though the L1
configuration does not fully eliminate data movement (NOC transfers
remain).

**Observation 4.** K_block=1 is 7% faster than K_block=8 (3.19 vs
3.44 ms). One possible explanation: K_block=1 generates more frequent,
smaller DMA transfers, which may create more overlap between the two
DM threads operating concurrently. However, without per-RISC-thread
cycle counts, we cannot confirm whether the difference originates in
DMA scheduling or compute.

### 3.3 Compiler-Generated K Loop: Requirements for Scale

The v3-compiler-k version places the full K dimension in one DFB fill
and relies on the compiler to generate the K reduction loop. This is
currently limited to problem sizes where M_block * K + K * N_block
tiles fit in L1 (precluding the 4096^3 benchmark shape).

For v3-compiler-k to operate at scale (large K that does not fit in a
single DFB fill), the following compiler and runtime support is needed:

1. **K splitting.** The compiler must decompose the K dimension into
   K_block (DFB fill size, DST-accumulated within each block) and
   K_num_blocks (outer loop, L1-accumulated across blocks). The user
   writes `out.store(a @ b)` with K_block-sized DFBs; the compiler
   generates the outer K_num_blocks loop with `pack_reconfig_l1_acc`
   guards. This requires the Python tracer to support for-loops in
   compute that generate `scf.for` (PR #446, `bnorris/block-expr`
   branch).

2. **DFB streaming for K blocks.** The data movement threads must
   deliver K_block-sized chunks per iteration, synchronized with the
   compute thread's K loop via CB semaphores. The DFB API supports this
   (reserve/push per block with `block_count=2`), but the compiler does
   not yet automatically generate the DM-side K loop from the compute's
   K reduction.

3. **Split DMA for the generated K loop.** The compiler-generated
   reader and writer must distribute A and B reads across both RISC
   cores, as in the v2-split-dma optimization.

## 4. Data Movement Analysis

### 4.1 DRAM Read Volume

The following analysis quantifies the total DRAM read traffic for the
4096x4096x4096 benchmark (128x128x128 tiles, M_block=N_block=8,
K_block=8, K_num_blocks=16). Each tile is 2 KB (bf16, 32x32). These
are calculated values, not runtime measurements.

**Per output block.** Each (m, n) output block requires:
- A blocks: K_num_blocks x M_block x K_block = 16 x 8 x 8 = 1024 tiles
- B blocks: K_num_blocks x K_block x N_block = 16 x 8 x 8 = 1024 tiles
- Total: 2048 tiles = 4 MB per output block

**Per core (130 cores, 256 output blocks total).**
Each core processes ~2 output blocks. Total reads per core: ~8 MB.

**tt-metal with K-direction alternation.** tt-metal's `dm_in0_sender`
flips the K iteration direction between consecutive N blocks (verified
by reading `dm_in0_sender.cpp`):

```
N_block 0:  K = 0, 1, 2, ..., 15    (forward)
N_block 1:  K = 15, 14, ..., 1, 0   (backward)
N_block 2:  K = 0, 1, 2, ..., 15    (forward)
```

When the M block is the same across consecutive N blocks (row-major
output traversal), the A block for the first K iteration of N_block 1
is the same as the last K iteration of N_block 0. The reader skips the
read (`reuse_block = true` in `dm_in0_sender.cpp:359`).

Over N_blocks_per_core consecutive N iterations with the same M block,
this eliminates (N_blocks_per_core - 1) A block reads out of
N_blocks_per_core x K_num_blocks total. For N_blocks_per_core = 2
(typical at 130 cores with 256 output blocks), this saves 1 out of 32
A block reads -- a 1.5% reduction in total read volume.

For configurations with more N blocks per core (fewer cores or larger
problems), the savings increase. With N_blocks_per_core = 16 (e.g.,
16 cores), 15 out of 256 A block reads per core are eliminated -- a 6%
reduction.

**tt-lang without K-direction alternation.** Every A and B block is
read from DRAM independently. No reuse across output blocks.

The read volume difference between tt-metal and tt-lang is 1.5% for the
130-core, 4096^3 configuration. This is too small to explain the
observed 3.5x slowdown.

### 4.2 DMA Scheduling

The following are known structural differences between the tt-lang and
ttnn.matmul data movement implementations. Their individual
contributions to the observed 3.5x slowdown have not been measured
independently (this would require per-RISC-thread cycle profiling).

1. **Double buffering.** Both DM threads use `block_count=2`. The CB
   semaphore protocol allows the reader to start the next block's DMA
   while compute processes the current block. We verified empirically
   that increasing `block_count` to 3 (triple buffering) produced no
   improvement (3.48 ms vs 3.43 ms for K_block=8, within noise). This
   indicates that the double-buffering pipeline depth is not the
   bottleneck, but does not tell us whether the pipeline is effectively
   utilized or whether DMA latency per block exceeds compute time per
   block (which would cause compute to stall regardless of buffer
   depth).

2. **K-direction alternation.** As quantified in Section 4.1, this
   saves 1.5% of total read volume at 130 cores. The performance
   effect, if any, is within measurement noise.

3. **Deferred output writes.** tt-metal defers the previous output
   block's DRAM write until a specific K iteration of the next block
   (`defer_write_k_block` in `dm_in0_sender.cpp:224`), overlapping
   output writes with input reads. tt-lang writes output synchronously
   after each output block completes. The performance impact has not
   been measured in isolation.

4. **Concurrent input reads.** With v2-split-dma, A and B reads
   execute on separate RISC cores concurrently. This is already
   implemented. The v1-single-reader vs v2-split-dma comparison
   (pending re-measurement) will quantify the improvement.

### 4.3 Expressibility in tt-lang

| Optimization | Status | Required infrastructure |
|-------------|--------|------------------------|
| Split DMA | Implemented (v2) | -- |
| Double buffering | Active (`block_count=2`) | -- |
| Concurrent A+B reads | Active (via split DMA) | -- |
| K-direction alternation | Not implemented | Runtime ternary expressions (`k_forward ? k : K-1-k`) |
| Deferred output writes | Not implemented | DM-thread coordination, runtime conditionals |
| Multicast input distribution | Not implemented | A subset of cores read from DRAM and multicast to neighbors, reducing total DRAM read traffic. Requires multicast NOC primitives in the DM kernel. |

## 5. Reproducing

```bash
# From the tt-lang root, inside the Docker container:
source build-docker/env/activate
python examples/matmul_bench/bench_matmul.py
```

Results are appended to `examples/matmul_bench/matmul_perf_results.csv`.

For per-line cycle annotation (requires profiler-enabled build):
```bash
TT_METAL_DEVICE_PROFILER=1 TTLANG_AUTO_PROFILE=1 \
  python examples/matmul_bench/bench_matmul.py
```

## 6. tt-metal Reference Configuration

The following is derived from reading the tt-metal source code
(`test_minimal_matmul.py`, `minimal_matmul/device/kernels/compute.cpp`,
`dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`), not from runtime
profiling of ttnn.matmul.

tt-metal `test_minimal_matmul.py::test_linear`:
- Shape: 4096 x 4096 x 4096
- Blocks: M=8, K=8, N=8, subblock_h=2, subblock_w=2
- Config: `fp32_dest_acc_en=True`, `packer_l1_acc=True`,
  `math_fidelity=HiFi2`
- Compute kernel uses L1 accumulation across K_num_blocks=16 with
  subblocked DST accumulation within each K_block
- DMA split across BRISC (in0 + relay) and NCRISC (in1 + output)
- K-direction alternation for in0 reuse
- Deferred output writes overlapped with next block's DMA
