# Compiler/emulator interface investigation

These are source-investigation notes written before the focused probes. See
[final triage](TRIAGE.md), [static-configuration evidence](NUMERICAL_STATIC_CONFIG_NOTES.md),
and [tree/loopback evidence](NUMERICAL_TREE_LOOPBACK_NOTES.md) for subsequent
results and current confidence assessments.

Source investigation for compiler `70c5d093df06d211aa78332447bbd8d602d2ba79`,
emulator `7292395ce55a208a8ede0a4635a9f2167c8c4939`, and Metal
`b6c508c4790fac0a11597e43a5421624cb461553`. These notes describe mechanisms;
the fresh JUnit reports and final TRIAGE.md determine the observed counts.

## DFB reset and reconfiguration

The TT-Lang implementation includes Blackhole RISC-V `lw`, `sw`, and
`and x0` instructions. The emulator compiles generated kernels for an x86-64
host. Its kernel patcher rewrites standalone `fence` instructions but not these
multi-instruction blocks. That prevents JIT compilation before execution.

Compiler evidence, relative to the checkout:

- `include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reset.h:42`
- `include/ttlang/Target/TTKernel/LLKs/experimental_dfb_reconfiguration.h:45`
- `lib/Target/TTKernel/TTKernelToCpp.cpp:285`: custom header bodies are embedded
  verbatim in generated code. A same-named shadow header alone cannot replace them.

Pinned emulator evidence, relative to its checkout:

- `include/tt_emule/detail/kernel_patcher.hpp:124`: standalone fence rewrite.
- Existing emulator kernel utility helpers provide cooperative waits and fused
  compute participation, with asymmetric barriers.
- `include/jit_hw/internal/cb_interface.h:12`: local CB geometry is insufficient;
  wrapping and advancement use shared CBSyncState.

Removing the assembly is not a complete fix. Hardware uses separate UNPACK
and PACK participants; emulation combines them in one compute worker. Plain
busy-spins can prevent cooperative progress. Reset completion ordering, shared
stream counters, shared geometry, and both low/high CB index masks must all be
preserved. The compiler's completion barrier is not automatically equivalent
to the existing emulator utility barrier.

Suggested implementation boundary: provide emulation implementations of
`experimental::complete_dfb_interface_work`, `reset_dfb_interfaces`, and
`reconfigure_dfb_interfaces`, selected for emulated kernels while retaining the
hardware implementations. Reuse L1 translation, canonical geometry rebinding,
view reseating, and stream reset helpers; do not turn the operation into a no-op.

Focused regressions: repeated reset; indices 32-63; preserving unselected live
buffers; tensor-backed storage switching; changed capacity; discard behavior;
and repeated cached-kernel execution.

## Fused RMSNorm

The immediate failure is the absent `ckernel_sfpu_rsqrt.h` required at
`include/ttlang/Target/TTKernel/LLKs/experimental_row_normalization.h:17`.
The emulator contains `sfpu/ckernel_sfpu_rsqrt_compat.h`, but that compatibility
forwarder does not implement the same complete operation.

Static inspection identifies further likely blockers beyond the first missing
header: custom `ckernel_template`, `TT_OP_ELWMUL`, `TT_OP_UNPACR`, and
`TTI_MOVD2B` programming in the compiler header (lines 95, 127, 244) lacks
matching implementations in the pinned emulator's shadow instruction layer.
Those additional failures have not been claimed as separately reproduced.

A whole-operation adapter for `experimental::row_normalization_block` is a
narrower option than implementing the full custom instruction-template engine.
Reusable emulator operations include `mul_reduce_scalar_tile`, `add_rsqrt_tile`,
and DST reuse helpers. Preserve optional gamma, broadcasts, tile geometry,
caller-owned DST results, and precision/rounding boundaries.

Do not substitute the existing `rmsnorm_compute_impl` directly: it requires
gamma and writes output CB memory, whereas the compiler operation leaves its
results in DST for later packing by its caller.

Focused regressions: materialized-reference comparison; no/full/broadcast gamma;
DRAM and L1; 16/32-row tiles; BF16/FP32 DST; half/full DST synchronization; and
the five-tile FP32 case.

## Device-print tests

The pinned emulator's `include/jit_hw/api/debug/dprint.h` forwards DPRINT to
DEVICE_PRINT. `include/jit_hw/api/debug/device_print.h:6-15` explicitly discards
DEVICE_PRINT and all processor-specific variants. FileCheck cannot find the
expected output. This is an observability feature gap, not evidence by itself
of wrong tensor computation. Implement the required print behavior, or explicitly
classify these as unsupported; do not report skipped cases as passes.

## Numerical mismatches: hypotheses awaiting diagnostic output

A concrete descriptor-selection discrepancy was found during source comparison.
The pinned runtime's
`tt_metal/impl/emulation/emulated_program_runner.cpp:1573-1679` builds the shared
kernel's CB format and geometry tables using only the first core in the kernel's
core range. A CB absent on that core receives format 255 (`Invalid`). The pinned
emulator's `include/jit_hw/api/compute/common.h:430` uses the format enum without
the page-size fallback mentioned by the runtime's stale comment; invalid format
passes its guard and reaches BF16 unpack/pack fallbacks (lines 1016 and 703).

Hardware descriptor collection differs. In the same pinned Metal source,
`tt_metal/impl/program/program.cpp:2749` calls `set_cb_data_fmt_and_tile` with all
kernel core ranges. Lines 2321-2333 visit each range and every intersecting CB;
the intersection test is in `tt_metal/impl/buffers/circular_buffer.cpp:100-102`.
Consequently, a consistently typed physical slot present on another executing
core is represented in the hardware descriptors even if absent on the root.
This supports emulator-side ownership of the descriptor inconsistency, subject
to confirming that the failing compiled programs have the suspected sparse slots.

Falsifiable prediction for the incompatible-static-configuration test: the first
input exists only on core (0,0), the broadcast input only on (1,0), and output on
both. If the second input is treated as BF16, the left exponential half remains
correct while every row of the right half becomes
`[0,0,0,1/32,0,2/32,...,0,7/32, then 16 zeros]`, giving exactly 992 incorrect
right-half elements. The post-failure probe captures cached CB allocation metadata
and tensors to check this without changing allocation or execution.

- Cross-DFB multicast loopback: four-core BF16 constant-7 transfer, including
  the sending core, with different source and receiver buffers. Both compiler
  lowering and emulator expose loopback paths, so "loopback unimplemented" is
  not an established cause. Inspect incorrect receiver/stripe locations and
  whether output retains the -42 sentinel or becomes zero.
- Default FP32 tree reduction in DRAM and L1: eight cores, L1-pack accumulation,
  14 output rows in a 16x32 tile. Compare specialized FP32 and default BF16
  controls. Possible fault boundaries include unspecialized per-core control
  flow, packing geometry, and accumulator initialization.
- Incompatible static DFB configurations in DRAM and L1: a two-node program
  computes FP32 exp on one node and row broadcast on the other. Determine whether
  the three-allocation assertion or the numerical assertion fails, then identify
  the wrong half of the output before assigning ownership.

Existing issues #924 and #1037 were inspected; they describe different
reproducers. Neither was assumed to explain these numerical failures.
