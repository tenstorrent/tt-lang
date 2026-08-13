// RUN: ttlang-opcost-report | FileCheck %s

// What the operation-cost table contains. `perf_data/` is not in the tree, so
// these numbers are the record of the measurement data: a regeneration that
// changes coverage changes this test, and the diff belongs in the same review as
// the data change. That churn is the point -- a sweep that silently drops
// measurements should not pass.
//
// The tool links TTLangOpCost alone, so this also proves the library stands up
// without MLIR or any dialect.

// CHECK: operations 296
// CHECK-NEXT: measured-rows 607

// Operations that occupy no engine: known to the table and costing nothing, as
// opposed to absent from it, which is unknown and should fail a caller.
// CHECK-NEXT: runs-nowhere 24

// Three counts per engine, and the gaps between them are the interesting part.
//
//   slots     -- (operation, engine) pairs the table could describe
//   measured  -- pairs any perf sweep reached
//   reachable -- pairs a bf16 -> bf16, dest_acc=off, 4-face kernel can resolve
//
// dm is zero because the LLK suite builds only TRISC kernels: nothing in it ever
// runs on NCRISC or BRISC, so every data-movement cost is a placeholder.
// CHECK-NEXT: engine dm slots 56 measured 0 reachable 0
// CHECK-NEXT: engine unpack slots 43 measured 9 reachable 8

// measured 50 against reachable 6 is not a bug in the lookup, and has two causes.
//
// Isolating an SFPU operation's math cost requires unpack_to_dest, which the
// hardware offers only for a 32-bit input with dest_acc on, so those rows exist
// only at Float32 and a bf16 kernel correctly fails to match them.
//
// reduce_tile is unreachable for a different reason: its cost depends on the
// reduce dimension and pool type, which are attributes of the operation rather
// than of the kernel -- math spans 19 to 133 cycles across the six combinations.
// KernelConfig is kernel-wide and has nowhere to put them, so the rows are keyed
// correctly and cannot yet be asked for. Averaging them into one number would be
// worse than the placeholder.
// CHECK-NEXT: engine math slots 208 measured 50 reachable 6
// CHECK-NEXT: engine pack slots 26 measured 4 reachable 4
// CHECK-NEXT: total slots 333 measured 63 reachable 18

// The outcomes a caller has to handle, one of each.
// CHECK-NEXT: lookup pack_tile/pack measured {{[0-9.]+}} per-tile
// CHECK-NEXT: lookup add_tiles/unpack measured {{[0-9.]+}} per-tile
// CHECK-NEXT: lookup matmul_block/math placeholder 400.00 per-call
// CHECK-NEXT: lookup pack_tile/math not-on-engine
// CHECK-NEXT: lookup get_compile_time_arg_val/math not-on-engine

// The whole reason for two entry points: matmul_block has measurements, but all
// of them are half-tile fits carrying an intercept, so the configuration above
// matches none. lookupOrPlaceholder invents a number, lookupMeasured refuses.
// A caller reporting an absolute figure wants the refusal.
// CHECK-NEXT: measured-only matmul_block/math none

// Knobs are a caller-supplied list rather than fixed fields because they come
// from three different places: kernel-wide settings, a circular buffer's format
// decision, and the operation's own attributes. A reduce needs the last kind --
// its math cost spans 19 to 133 cycles across reduce dimension and pool type, so
// no kernel-wide value can answer for it. Supplying them turns an unmatchable
// row into the measurement.
// CHECK-NEXT: reduce without op attrs none
// CHECK-NEXT: reduce with op attrs measured 19.19

// Costs are not transferable across architectures, so one with no table answers
// nothing rather than borrowing Blackhole's.
// CHECK-NEXT: wormhole operations 0
// CHECK-NEXT: wormhole known-op pack_tile no
