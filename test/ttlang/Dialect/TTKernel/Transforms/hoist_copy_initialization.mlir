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

// -----

// Matching source setup moves together before the loop.
// CHECK-LABEL: func.func @reconfigured_copy
// CHECK: ttkernel.reconfig_data_format(%[[SOURCE:.*]], %[[SOURCE]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[SOURCE]])
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: return
func.func @reconfigured_copy(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Repeated matching setup pairs share one source configuration and init.
// CHECK-LABEL: func.func @repeated_reconfigured_copy
// CHECK: ttkernel.reconfig_data_format(%[[SOURCE:.*]], %[[SOURCE]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[SOURCE]])
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: return
func.func @repeated_reconfigured_copy(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Packer accumulation leaves the unpack/math copy configuration intact.
// CHECK-LABEL: func.func @l1_pack_preserves_copy
// CHECK: ttkernel.reconfig_data_format(%[[SOURCE:.*]], %[[SOURCE]])
// CHECK-NEXT: ttkernel.copy_tile_init(%[[SOURCE]])
// CHECK-NEXT: scf.for
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: return
func.func @l1_pack_preserves_copy(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.pack_reconfig_l1_acc(%enabled) : (i32) -> ()
  }
  return
}

// -----

// An empty loop does not execute source reconfiguration early.
// CHECK-LABEL: func.func @reconfigured_empty_loop
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func @reconfigured_empty_loop(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %zero step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Unknown trip counts retain the complete setup pair inside the loop.
// CHECK-LABEL: func.func @reconfigured_dynamic_loop
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func @reconfigured_dynamic_loop(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %limit step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// Both physical sources must match the copy source.
// CHECK-LABEL: func.func @mismatched_source_pair
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func @mismatched_source_pair(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %other) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// A trailing source reconfiguration may reset signed-zero state.
// CHECK-LABEL: func.func @standalone_reconfiguration
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func @standalone_reconfiguration(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
  }
  return
}

// -----

// Later format setup must not change formats used by an earlier copy.
// CHECK-LABEL: func.func @late_reconfiguration
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func @late_reconfiguration(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}

// -----

// An unknown call may clobber the hoisted source configuration.
// CHECK-LABEL: func.func @reconfigured_unknown_call
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: return
func.func private @unknown()
func.func @reconfigured_unknown_call(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    func.call @unknown() : () -> ()
  }
  return
}

// -----

// A second setup pair for a different source prevents hoisting.
// CHECK-LABEL: func.func @reconfigured_different_source
// CHECK-NOT: ttkernel.reconfig_data_format
// CHECK-NOT: ttkernel.copy_tile_init
// CHECK: scf.for
// CHECK: ttkernel.reconfig_data_format
// CHECK: ttkernel.copy_tile_init
// CHECK: return
func.func @reconfigured_different_source(%source: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %other: !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, %limit: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %enabled = arith.constant 1 : i32
  scf.for %iteration = %zero to %four step %one {
    ttkernel.reconfig_data_format(%source, %source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%source) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%source, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
    ttkernel.reconfig_data_format(%other, %other) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile_init(%other) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>) -> ()
    ttkernel.copy_tile(%other, %zero, %zero) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index, index) -> ()
  }
  return
}
