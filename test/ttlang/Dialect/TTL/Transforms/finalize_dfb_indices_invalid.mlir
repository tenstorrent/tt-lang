// Invalid control-flow and lifetime contracts for ttl-finalize-dfb-indices.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'

// A resident loop needs a forward phase cut and a cyclic return cut.

// expected-error @below {{a cyclic resident loop requires at least two reset_dataflow_buffers calls}}
func.func @one_cyclic_cut()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// A reset under conditional control flow cannot define one static phase order.

func.func @conditional_reset(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %condition {
      // expected-error @below {{must be top-level or a direct child of one top-level resident scf.for loop}}
      ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
      // expected-error @below {{must be top-level or a direct child of one top-level resident scf.for loop}}
      ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    }
  }
  return
}

// -----

// A nested loop is not the one canonical top-level resident loop.

func.func @nested_loop_reset()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %outer = %c0 to %c4 step %c1 {
    scf.for %inner = %c0 to %c4 step %c1 {
      // expected-error @below {{must be top-level or a direct child of one top-level resident scf.for loop}}
      ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
      // expected-error @below {{must be top-level or a direct child of one top-level resident scf.for loop}}
      ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    }
  }
  return
}

// -----

// Reset calls split between resident loops have an ambiguous cyclic order.

func.func @resets_in_different_loops()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  scf.for %i = %c0 to %c4 step %c1 {
    // expected-error @below {{must be top-level or a direct child of one top-level resident scf.for loop}}
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// Every kernel thread must use the same linear or cyclic reset shape.

func.func @linear_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  return
}

// expected-error @below {{must place reset_dataflow_buffers in the same control flow shape as every other kernel thread}}
func.func @cyclic_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// One logical DFB cannot belong to different phases on different threads.

func.func @phase_a_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "foreign_cb_access"(%cb0) {header = "foreign.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

func.func @phase_b_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    // expected-error @below {{dataflow buffer 0 is used across a reset_dataflow_buffers boundary}}
    ttl.opaque_call "foreign_cb_access"(%cb0) {header = "foreign.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// A reset ordinal has one card-wide preserve set.

func.func @preserve_zero_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "ttlang::reset_dataflow_buffers"(%cb0) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

func.func @preserve_one_thread()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // expected-error @below {{must preserve the same dataflow buffers at reset ordinal 0 as every other kernel thread}}
  ttl.opaque_call "ttlang::reset_dataflow_buffers"(%cb1) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}
