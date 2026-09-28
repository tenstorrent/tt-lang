// Verifies that payload-completion metadata does not change Metal DFB calls.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp %t.emitc.mlir | FileCheck %s --check-prefix=CPP

module {
  // CHECK-LABEL: func.func @release
  // CHECK: emitc.verbatim "cb_ctarg_0.push_back({});"
  // CHECK: emitc.verbatim "cb_ctarg_0.pop_front({});"
  // CHECK-NOT: push_back<true>
  // CHECK-NOT: pop_front<true>
  // CPP: cb_ctarg_0.push_back({{.*}});
  // CPP: cb_ctarg_0.pop_front({{.*}});
  // CPP-NOT: push_back<true>
  // CPP-NOT: pop_front<true>
  func.func @release() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %storage = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<2, !ttcore.tile<32x32, f32>>
    %one = arith.constant 1 : i32
    ttkernel.cb_push_back(%storage, %one) {payload_complete}
        : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_pop_front(%storage, %one) {payload_complete}
        : (!ttkernel.cb<2, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }
}
