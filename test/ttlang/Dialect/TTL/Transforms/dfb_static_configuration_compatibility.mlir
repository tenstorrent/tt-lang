// Tests static compute-configuration constraints on physical DFB reuse.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=NO-REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true},ttl-set-compute-kernel-config)' | FileCheck %s --check-prefix=CONFIG

// Disjoint launch-node domains do not permit f32 DFBs with incompatible
// unpack modes to share one physical index.

// REUSE-LABEL: func.func @disjoint_incompatible_configuration
// REUSE: %[[SFPU:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: %[[BCAST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

// REPORT: DFB conflict lhs=0 rhs=1 reason=static-configuration-mismatch

// NO-REUSE-LABEL: func.func @disjoint_incompatible_configuration
// NO-REUSE: %[[SFPU:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// NO-REUSE-NEXT: %[[BCAST:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @disjoint_incompatible_configuration()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %sfpu_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %bcast_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %node_x = ttl.core_x : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    scf.if %first_node {
      %sfpu_wait = ttl.cb_wait %sfpu_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %sfpu_attached = ttl.attach_cb %sfpu_wait, %sfpu_dfb
          : (tensor<1x1x!ttcore.tile<32x32, f32>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %sfpu_tile = tensor.extract %sfpu_attached[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
      %exponential = ttl.tile_exp %sfpu_tile into dst[%zero]
          : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.cb_pop %sfpu_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    %second_node = arith.cmpi eq, %node_x, %one : index
    scf.if %second_node {
      %bcast_wait = ttl.cb_wait %bcast_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %bcast_attached = ttl.attach_cb %bcast_wait, %bcast_dfb
          : (tensor<1x1x!ttcore.tile<32x32, f32>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %bcast_tile = tensor.extract %bcast_attached[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
      %broadcast = ttl.tile_bcast %bcast_tile, %bcast_tile 1 : i32
          into dst[%zero]
          : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
            -> !ttcore.tile<32x32, f32>
      ttl.cb_pop %bcast_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    return
  }
}

// -----

// Disjoint launch-node domains still permit DFBs with identical static
// compute configurations to share one physical index.

// REUSE-LABEL: func.func @disjoint_compatible_configuration
// REUSE: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// REUSE-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 3 : index}

// REPORT-NOT: DFB conflict lhs=2 rhs=3 reason=static-configuration-mismatch

// NO-REUSE-LABEL: func.func @disjoint_compatible_configuration
// NO-REUSE: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 2 : index}
// NO-REUSE-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 3 : index}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @disjoint_compatible_configuration()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %node_x = ttl.core_x : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    scf.if %first_node {
      %first_wait = ttl.cb_wait %first_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %first_attached = ttl.attach_cb %first_wait, %first_dfb
          : (tensor<1x1x!ttcore.tile<32x32, f32>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %first_tile = tensor.extract %first_attached[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
      %first_exp = ttl.tile_exp %first_tile into dst[%zero]
          : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.cb_pop %first_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    %second_node = arith.cmpi eq, %node_x, %one : index
    scf.if %second_node {
      %second_wait = ttl.cb_wait %second_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %second_attached = ttl.attach_cb %second_wait, %second_dfb
          : (tensor<1x1x!ttcore.tile<32x32, f32>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %second_tile = tensor.extract %second_attached[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
      %second_exp = ttl.tile_exp %second_tile into dst[%zero]
          : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.cb_pop %second_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    return
  }
}

// -----

// An incompatible operation in a zero-trip loop does not constrain a physical
// DFB used by active operations.

// REUSE-LABEL: func.func @exact_empty_configuration_is_ignored
// REUSE: %[[ACTIVE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 4 : index}
// REUSE-NEXT: %[[INACTIVE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 5 : index}

// CONFIG-LABEL: func.func @exact_empty_configuration_is_ignored
// CONFIG: ttl.tile_add {{.*}}ttl.tile_execution_strategy = #ttl.tile_execution_strategy<fpu>

// REPORT-NOT: DFB conflict lhs=4 rhs=5 reason=static-configuration-mismatch

// NO-REUSE-LABEL: func.func @exact_empty_configuration_is_ignored
// NO-REUSE: %[[ACTIVE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 4 : index}
// NO-REUSE-NEXT: %[[INACTIVE:.*]] = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 5 : index}

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @exact_empty_configuration_is_ignored()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %active_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %inactive_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %active_wait = ttl.cb_wait %active_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %active_attached = ttl.attach_cb %active_wait, %active_dfb
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %active_tile = tensor.extract %active_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    %active_exp = ttl.tile_exp %active_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.cb_pop %active_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %node_x = ttl.core_x : index
    %outside_launch_grid = arith.cmpi eq, %node_x, %two : index
    scf.if %outside_launch_grid {
      %inactive_wait = ttl.cb_wait %inactive_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %inactive_attached = ttl.attach_cb %inactive_wait, %inactive_dfb
          : (tensor<1x1x!ttcore.tile<32x32, f32>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %inactive_tile = tensor.extract %inactive_attached[%zero, %zero]
          : tensor<1x1x!ttcore.tile<32x32, f32>>
      %inactive_add = ttl.tile_add %inactive_tile, %inactive_tile into dst[%zero]
          : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>
            -> !ttcore.tile<32x32, f32>
      ttl.cb_pop %inactive_dfb
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    return
  }
}
