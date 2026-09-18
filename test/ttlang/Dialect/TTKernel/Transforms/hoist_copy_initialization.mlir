// Checks copy initialization sharing and conservative hardware-state preservation.
// RUN: ttlang-opt %s --ttkernel-cleanup --split-input-file | FileCheck %s

// An empty loop must not introduce hardware initialization before returning.
// CHECK-LABEL: func.func @empty_loop
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @empty_loop(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %iteration = %zero to %zero step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// One initialization serves every copy in a statically nonempty loop.
// CHECK-LABEL: func.func @invariant_copy
// CHECK: ttkernel.copy_tile_init
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: return
func.func @invariant_copy(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Repeated identical initialization is redundant when intervening copies preserve configuration.
// CHECK-LABEL: func.func @duplicate_initializations
// CHECK: ttkernel.copy_tile_init
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: return
func.func @duplicate_initializations(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// A possibly empty loop must not configure hardware before its execution is known.
// CHECK-LABEL: func.func @dynamic_trip_count
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @dynamic_trip_count(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %limit step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// An opaque call may replace the copy configuration.
// CHECK-LABEL: func.func @unknown_call
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func private @unknown()
func.func @unknown_call(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    func.call @unknown() : () -> ()
  }
  return
}

// -----

// Changing the configured source between copies prevents sharing initialization.
// CHECK-LABEL: func.func @different_source
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @different_source(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile_init(%other) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%other, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Another initialization changes the unpack/math configuration used by the next iteration.
// CHECK-LABEL: func.func @other_configuration
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @other_configuration(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.init_sfpu(%source, %other) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
  }
  return
}

// -----

// Conditional initialization requires a separate execution proof and remains inside the loop.
// CHECK-LABEL: func.func @nested_control_flow
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @nested_control_flow(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    scf.if %condition {
      ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
      ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    }
  }
  return
}

// -----

// Moving a later initialization must not change the configuration used by an earlier copy.
// CHECK-LABEL: func.func @copy_before_init
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @copy_before_init(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index, %condition: i1) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}
