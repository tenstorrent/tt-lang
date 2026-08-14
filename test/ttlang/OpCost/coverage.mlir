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
// CHECK-NEXT: measured-rows 1938

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
// unpack gained add/sub/mul_tiles_init. Their init zone nests rather than
// partitions -- `binary_op_init_common` issues both of its calls and
// `binary_tiles_init` only the second -- so measuring the zone whole and
// measuring its tail attributes both owners from one pair of runs, with the
// whole still charged to `binary_op_init_common`. All three read 66.0 cycles,
// as they must: the eltwise type reaches only the math call, so they issue an
// identical `llk_unpack_AB_init`.
//
// The SFPU sources are deliberately not split this way. There `init_sfpu` issues
// both calls and the operation's own init is MATH(...) only, so there is no
// second owner to separate and a split would just halve one operation's cost.
// CHECK-NEXT: engine unpack slots 43 measured 13 reachable 11

// measured 110 against reachable 96. The gap used to be far wider: isolating an
// SFPU operation's math cost requires unpack_to_dest, which the hardware offers
// only for a 32-bit input with dest_acc on, so every clean SFPU row was Float32
// and no bf16 kernel could match one.
//
// Those are now recovered by subtraction. The SFPU sources measure their math
// zone twice -- whole, and with the SFPU call elided so only the datacopy remains
// -- in the same kernel and build, so the pipeline fill cancels. Checked against
// the one configuration where direct isolation does work: 218.9 by subtraction
// against 216.8 measured directly, within 1%.
//
// reduce_tile is unreachable for a different reason: its cost depends on the
// reduce dimension and pool type, which are attributes of the operation rather
// than of the kernel -- math spans 19 to 133 cycles across the six combinations.
// KernelConfig is kernel-wide and has nowhere to put them, so the rows are keyed
// correctly and cannot yet be asked for. Averaging them into one number would be
// worse than the placeholder.
// CHECK-NEXT: engine math slots 208 measured 110 reachable 96
// CHECK-NEXT: engine pack slots 26 measured 5 reachable 4
// CHECK-NEXT: total slots 333 measured 128 reachable 111

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
// The value is matched loosely: it is a hardware measurement and moves by a
// fraction of a cycle between sweeps. What matters is that supplying the
// operation's attributes turns an unmatchable row into a measured one.
// CHECK-NEXT: reduce with op attrs measured 19.{{[0-9]+}}

// Costs are not transferable across architectures, so one with no table answers
// nothing rather than borrowing Blackhole's.
// CHECK-NEXT: wormhole operations 0
// CHECK-NEXT: wormhole known-op pack_tile no
