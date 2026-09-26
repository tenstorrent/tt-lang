// A scalar print lowered by TT-Lang retains its marker and is accepted by the
// compiler-managed SRAM validator.
// RUN: ttlang-opt %s --convert-ttkernel-to-emitc | FileCheck %s

module attributes {ttl.memory_model = "compiler-sram", ttl.l1_arena_bytes = 0 : i64, ttl.dfb_allocations = []} {
  // CHECK-LABEL: func.func @scalar_print
  // CHECK: emitc.verbatim "ttmlir::dprint(1);" {ttl.dprint_generated}
  func.func @scalar_print() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %unused = ttkernel.get_compile_time_arg_val(0) : () -> i32
    emitc.verbatim "ttmlir::dprint(1);" {ttl.dprint_generated}
    return
  }
}
