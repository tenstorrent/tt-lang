// RUN: ttlang-opt --pass-pipeline='builtin.module(ttkernel-cost-estimate{detail=1})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKEYED
// RUN: ttlang-opt --pass-pipeline='builtin.module(ttkernel-cost-estimate{detail=1 math-fidelity=HiFi4})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=HIFI4
// RUN: ttlang-opt --pass-pipeline='builtin.module(ttkernel-cost-estimate{detail=1 math-fidelity=LoFi})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOFI

// The knobs a measurement can be keyed on, and where each one comes from.
//
// TTLangOpCost keys its rows on more than the operation and the data format,
// and it cannot guess: a row naming a knob the caller did not answer is
// unmatchable. This kernel exercises all three origins in one function -- the
// reduce attributes (`mathop`, `reduce_pool_type`), the SFPU exponential's own
// attributes (`approx_mode`, `iterations`, `input_clamping`), and math fidelity,
// which the IR does not carry and the pass option supplies.
//
// The costs below are the table's own numbers, so a lookup that stops matching
// shows up as a cost changing rather than as a column no one reads. They move
// when the table is regenerated -- read the new ones out of
// lib/Analysis/CostTableBlackhole.inc rather than off this test's failure, since
// the point of each is which row it came from.

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {
      dst_full_sync_en = false,
      fp32_dest_acc_en = false,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %cb2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>

    ttkernel.tile_regs_acquire() : () -> ()

    // Reduce over rows, summing: the pair of attributes the table calls
    // `mathop` and `reduce_pool_type`. Both halves of the reduce read them, and
    // the math half is keyed on fidelity as well.
    ttkernel.reduce_init(%cb0, %cb1, %cb2, <reduce_sum>, <reduce_dim_row>) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.reduce_tile(%cb0, %cb1, %c0, %c0, %c0, <reduce_sum>, <reduce_dim_row>) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()

    // An FPU multiply, whose math cost spans a factor of four across the four
    // fidelities and whose unpack cost does not depend on them at all.
    ttkernel.mul_tiles_init(%cb0, %cb1) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.mul_tiles(%cb0, %cb1, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()

    // The approximate exponential, at metal's default trip count and default
    // clamping. Fidelity does not reach the SFPU, so these match whether or not
    // it was supplied.
    ttkernel.exp_tile_init() {approx = true} : () -> ()
    ttkernel.exp_tile(%c0) {approx = true, iterations = 8 : i32} : (index) -> ()

    // The same approximate exponential with the clamp turned off, which is the
    // one flag on this operation worth a factor of four: skipping the check on
    // very negative inputs takes the tile from 112 cycles to 29.
    ttkernel.exp_tile_init() {approx = true, input_clamping = #ttkernel.input_clamping<none>} : () -> ()
    ttkernel.exp_tile(%c0) {approx = true, iterations = 8 : i32, input_clamping = #ttkernel.input_clamping<none>} : (index) -> ()

    // The same operation with no attributes at all, which is what tt-lang emits
    // for a plain `ttl.exp`: the Python wrapper drops every flag left at its
    // default, so the op lowers to a bare `exp_tile(idst)` compiled with metal's
    // approx=false and iterations=8. The knobs have to answer those defaults or
    // the ordinary case is the one that misses.
    ttkernel.exp_tile_init() : () -> ()
    ttkernel.exp_tile(%c0) : (index) -> ()

    ttkernel.tile_regs_commit() : () -> ()
    ttkernel.tile_regs_wait() : () -> ()
    ttkernel.pack_tile(%c0, %cb2, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
    ttkernel.tile_regs_release() : () -> ()
    return
  }
}

// Without fidelity the three rows keyed on it are unmatched rather than
// answered at some other fidelity's cost, and they are counted apart from the
// four DST handshakes, which no sweep timed at all.
// UNKEYED: cost estimate: 12 of 20 placements measured
// UNKEYED-NEXT: 4 unmatched {{.*}}, 4 untimed
// HIFI4: cost estimate: 16 of 20 placements measured
// HIFI4-NEXT: 0 unmatched {{.*}}, 4 untimed
// LOFI: cost estimate: 16 of 20 placements measured
// LOFI-NEXT: 0 unmatched {{.*}}, 4 untimed

// The unpack halves do not depend on fidelity, so all three runs agree on them.
// Both reduce halves need `mathop` and `reduce_pool_type`: recovered here from
// reduce_dim and reduce_type, and unanswerable from the operation's name alone.
// UNKEYED: TRISC0 unpack
// UNKEYED: ttkernel.reduce_init {{.*}} 72 {{.*}} meas
// UNKEYED: ttkernel.reduce_tile {{.*}} 37 {{.*}} meas
// UNKEYED: ttkernel.mul_tiles {{.*}} 37 {{.*}} meas
// HIFI4: TRISC0 unpack
// HIFI4: ttkernel.reduce_init {{.*}} 72 {{.*}} meas
// HIFI4: ttkernel.reduce_tile {{.*}} 37 {{.*}} meas
// HIFI4: ttkernel.mul_tiles {{.*}} 37 {{.*}} meas
// LOFI: TRISC0 unpack
// LOFI: ttkernel.reduce_init {{.*}} 72 {{.*}} meas
// LOFI: ttkernel.reduce_tile {{.*}} 37 {{.*}} meas
// LOFI: ttkernel.mul_tiles {{.*}} 37 {{.*}} meas

// Math is where fidelity lands, for all four of this kernel's math rows.
// `nokey` says a measurement exists that this kernel cannot key, which supplying
// the fidelity closes; the rows then carry LoFi's numbers, a third of HiFi4's for
// the multiply.
// UNKEYED: TRISC1 math
// UNKEYED: ttkernel.reduce_init {{.*}} nokey
// UNKEYED: ttkernel.reduce_tile {{.*}} nokey
// UNKEYED: ttkernel.mul_tiles_init {{.*}} nokey
// UNKEYED: ttkernel.mul_tiles {{.*}} nokey
// HIFI4: TRISC1 math
// HIFI4: ttkernel.reduce_init {{.*}} 134 {{.*}} meas
// HIFI4: ttkernel.reduce_tile {{.*}} 52 {{.*}} meas
// HIFI4: ttkernel.mul_tiles_init {{.*}} 86 {{.*}} meas
// HIFI4: ttkernel.mul_tiles {{.*}} 84 {{.*}} meas
// LOFI: TRISC1 math
// LOFI: ttkernel.reduce_init {{.*}} 133 {{.*}} meas
// LOFI: ttkernel.reduce_tile {{.*}} 15 {{.*}} meas
// LOFI: ttkernel.mul_tiles_init {{.*}} 87 {{.*}} meas
// LOFI: ttkernel.mul_tiles {{.*}} 19 {{.*}} meas

// The exponential's own attributes reach the table, and its cost does not depend
// on fidelity: approximate at metal's default trip count is 29 against the exact
// path's 152. It names no circular buffer at all, so the format it was keyed on
// comes from the kernel's buffers, which here all hold bf16.
//
// Three shapes, and the middle one is why the clamp has to be answered: the same
// approximate exponential costs 112 clamped and 29 with the check skipped, and
// its init moves the other way, 85 against 132. The
// last pair carries no attributes at all, which is what tt-lang emits for a plain
// `ttl.exp` -- the Python wrapper drops every flag left at its default -- so the
// knobs answer metal's defaults on its behalf and it lands on the exact,
// clamped path at 73 and 152.
// UNKEYED: ttkernel.exp_tile_init {{.*}} 85 {{.*}} meas
// UNKEYED-NEXT: ttkernel.exp_tile {{.*}} 112 {{.*}} meas
// UNKEYED-NEXT: ttkernel.exp_tile_init {{.*}} 132 {{.*}} meas
// UNKEYED-NEXT: ttkernel.exp_tile {{.*}} 29 {{.*}} meas
// UNKEYED-NEXT: ttkernel.exp_tile_init {{.*}} 73 {{.*}} meas
// UNKEYED-NEXT: ttkernel.exp_tile {{.*}} 152 {{.*}} meas
// HIFI4: ttkernel.exp_tile_init {{.*}} 85 {{.*}} meas
// HIFI4-NEXT: ttkernel.exp_tile {{.*}} 112 {{.*}} meas
// HIFI4-NEXT: ttkernel.exp_tile_init {{.*}} 132 {{.*}} meas
// HIFI4-NEXT: ttkernel.exp_tile {{.*}} 29 {{.*}} meas
// HIFI4-NEXT: ttkernel.exp_tile_init {{.*}} 73 {{.*}} meas
// HIFI4-NEXT: ttkernel.exp_tile {{.*}} 152 {{.*}} meas
// LOFI: ttkernel.exp_tile_init {{.*}} 85 {{.*}} meas
// LOFI-NEXT: ttkernel.exp_tile {{.*}} 112 {{.*}} meas
// LOFI-NEXT: ttkernel.exp_tile_init {{.*}} 132 {{.*}} meas
// LOFI-NEXT: ttkernel.exp_tile {{.*}} 29 {{.*}} meas
// LOFI-NEXT: ttkernel.exp_tile_init {{.*}} 73 {{.*}} meas
// LOFI-NEXT: ttkernel.exp_tile {{.*}} 152 {{.*}} meas

// The DST handshakes are the untimed four: no perf source isolates a handshake,
// so they are modelled as pure synchronization and charged nothing.
// HIFI4: ttkernel.tile_regs_commit {{.*}} untimed
