// RUN: ttlang-opt --convert-ttkernel-to-emitc --split-input-file -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir 2>&1 | FileCheck %s --check-prefix=CPP

// Subsystem-level coverage for `memref<1xi32>` alloca + zero-store + load
// / add 1 / store through `--convert-ttkernel-to-emitc` and
// `ttlang-translate --ttkernel-to-cpp`. This is how the per-PipeNet
// receiver counter (issue #505) is materialized; isolating it here means
// a MemRefToEmitC regression surfaces locally instead of as a pipe
// lowering failure.

//===----------------------------------------------------------------------===//
// Single-counter case: one PipeNet, one Pipe→CB receive at a receiver.
//===----------------------------------------------------------------------===//

// EMITC-LABEL: func.func @one_counter
// ConvertAlloca emits emitc.variable with an empty initializer; the
// explicit zero-store is required (without it the counter would start
// with stack garbage).
// EMITC: %[[CTR1:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
// EMITC: %[[IDX0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// EMITC: %[[ZERO:.*]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
// EMITC: %[[INIT_LV:.*]] = emitc.subscript %[[CTR1]][%[[IDX0]]] : (!emitc.array<1xi32>, !emitc.size_t) -> !emitc.lvalue<i32>
// EMITC: emitc.assign %[[ZERO]] : i32 to %[[INIT_LV]] : <i32>

// Read-modify-write must reference the same %[[CTR1]] on both sides.
// EMITC: %[[LD_LV:.*]] = emitc.subscript %[[CTR1]][%[[IDX0]]] : (!emitc.array<1xi32>, !emitc.size_t) -> !emitc.lvalue<i32>
// EMITC: %[[V:.*]] = emitc.load %[[LD_LV]] : <i32>
// EMITC: %[[ONE:.*]] = "emitc.constant"() <{value = 1 : i32}> : () -> i32
// EMITC: %[[SUM:.*]] = emitc.add
// EMITC: %[[ST_LV:.*]] = emitc.subscript %[[CTR1]][%[[IDX0]]] : (!emitc.array<1xi32>, !emitc.size_t) -> !emitc.lvalue<i32>
// EMITC: emitc.assign %{{.*}} : i32 to %[[ST_LV]] : <i32>

// Same array name across init store, load, and increment store.
// CPP-LABEL: void kernel_main()
// CPP: int32_t [[CTR1_C:v[0-9]+]][1];
// CPP: size_t [[IDX0_C:v[0-9]+]] = 0;
// CPP: int32_t [[ZERO_C:v[0-9]+]] = 0;
// CPP: [[CTR1_C]][[[IDX0_C]]] = [[ZERO_C]];
// CPP: int32_t [[V_C:v[0-9]+]] = [[CTR1_C]][[[IDX0_C]]];
// CPP: int32_t [[ONE_C:v[0-9]+]] = 1;
// CPP: [[CTR1_C]][[[IDX0_C]]] = {{.*}};
func.func @one_counter() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %counter = memref.alloca() : memref<1xi32>
  %c0_idx = arith.constant 0 : index
  %i32_zero = arith.constant 0 : i32
  memref.store %i32_zero, %counter[%c0_idx] : memref<1xi32>

  %v = memref.load %counter[%c0_idx] : memref<1xi32>
  %c1_i32 = arith.constant 1 : i32
  %new = arith.addi %v, %c1_i32 : i32
  memref.store %new, %counter[%c0_idx] : memref<1xi32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Two counters in one function get distinct emitc.variables; load/store
// must not alias.
//===----------------------------------------------------------------------===//

// EMITC-LABEL: func.func @two_counters
// EMITC: %[[CA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
// EMITC: %[[CB:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
// EMITC: emitc.subscript %[[CA]][
// EMITC: emitc.assign
// EMITC: emitc.subscript %[[CB]][
// EMITC: emitc.assign
// EMITC: emitc.subscript %[[CA]][
// EMITC: emitc.load
// EMITC: emitc.add
// EMITC: emitc.subscript %[[CA]][
// EMITC: emitc.assign
// EMITC: emitc.subscript %[[CB]][
// EMITC: emitc.load
// EMITC: emitc.add
// EMITC: emitc.subscript %[[CB]][
// EMITC: emitc.assign

// CPP-LABEL: void kernel_main()
// CPP: int32_t [[CA_C:v[0-9]+]][1];
// CPP: int32_t [[CB_C:v[0-9]+]][1];
// CPP-NOT: int32_t v{{[0-9]+}}[1];
func.func @two_counters() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %ctr_a = memref.alloca() : memref<1xi32>
  %ctr_b = memref.alloca() : memref<1xi32>
  %c0_idx = arith.constant 0 : index
  %i32_zero = arith.constant 0 : i32
  memref.store %i32_zero, %ctr_a[%c0_idx] : memref<1xi32>
  memref.store %i32_zero, %ctr_b[%c0_idx] : memref<1xi32>

  %va = memref.load %ctr_a[%c0_idx] : memref<1xi32>
  %c1_i32 = arith.constant 1 : i32
  %na = arith.addi %va, %c1_i32 : i32
  memref.store %na, %ctr_a[%c0_idx] : memref<1xi32>

  %vb = memref.load %ctr_b[%c0_idx] : memref<1xi32>
  %nb = arith.addi %vb, %c1_i32 : i32
  memref.store %nb, %ctr_b[%c0_idx] : memref<1xi32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Alloca at function entry, load-add-store inside an scf.if (the `if_dst`
// block pattern). The same array reference must reach inside the
// if-region (rewritten to emitc.if).
//===----------------------------------------------------------------------===//

// EMITC-LABEL: func.func @counter_in_if
// EMITC: %[[CTR3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<1xi32>
// EMITC: emitc.subscript %[[CTR3]][
// EMITC: emitc.assign
// EMITC: emitc.if
// EMITC: subscript %[[CTR3]][
// EMITC: load
// EMITC: add
// EMITC: subscript %[[CTR3]][
// EMITC: assign

// CPP-LABEL: void kernel_main()
// CPP: int32_t [[CTR3_C:v[0-9]+]][1];
// CPP: [[CTR3_C]][{{.*}}] = {{.*}};
// CPP: if ({{.*}}) {
// CPP: [[CTR3_C]][{{.*}}]
// CPP: [[CTR3_C]][{{.*}}] = {{.*}};
// CPP: }
func.func @counter_in_if() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %counter = memref.alloca() : memref<1xi32>
  %c0_idx = arith.constant 0 : index
  %i32_zero = arith.constant 0 : i32
  memref.store %i32_zero, %counter[%c0_idx] : memref<1xi32>

  // Condition computed in-function; mirrors how the receiver's `if_dst`
  // is lowered — `notSender` and similar predicates are arith-derived.
  %x = "ttkernel.my_logical_x_"() : () -> index
  %srcX = arith.constant 0 : index
  %cond = arith.cmpi ne, %x, %srcX : index
  scf.if %cond {
    %v = memref.load %counter[%c0_idx] : memref<1xi32>
    %c1_i32 = arith.constant 1 : i32
    %new = arith.addi %v, %c1_i32 : i32
    memref.store %new, %counter[%c0_idx] : memref<1xi32>
  }
  return
}
