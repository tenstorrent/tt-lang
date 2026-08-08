// Tests physical-index conflict graphs from launch-node-local DFB lifetimes.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// Each DFB conflicts with its two neighbors in a five-DFB cycle. Non-neighbors
// have disjoint launch domains and may share. At most two DFBs conflict
// pairwise, which proves only a two-index lower bound, but alternating two
// indices cannot close an odd cycle. The required third index demonstrates why
// the lower bound alone cannot decide every allocation.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}, {{.*}}dfb_index = 1 : i32{{.*}}, {{.*}}dfb_index = 2 : i32{{.*}}]}
// CHECK-LABEL: func.func @cycle_vertex_zero
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 50 : index}
// CHECK-LABEL: func.func @cycle_vertex_one
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 51 : index}
// CHECK-LABEL: func.func @cycle_vertex_two
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 52 : index}
// CHECK-LABEL: func.func @cycle_vertex_three
// CHECK: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 53 : index}
// CHECK-LABEL: func.func @cycle_vertex_four
// CHECK: ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 54 : index}

module attributes {ttl.launch_grid = [5 : i64, 1 : i64]} {
  func.func @cycle_vertex_zero()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 50 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %is_zero = arith.cmpi eq, %core_x, %zero : index
    %is_four = arith.cmpi eq, %core_x, %four : index
    %is_active = arith.ori %is_zero, %is_four : i1
    scf.if %is_active {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @cycle_vertex_one()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 51 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %is_zero = arith.cmpi eq, %core_x, %zero : index
    %is_one = arith.cmpi eq, %core_x, %one : index
    %is_active = arith.ori %is_zero, %is_one : i1
    scf.if %is_active {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @cycle_vertex_two()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 52 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %is_one = arith.cmpi eq, %core_x, %one : index
    %is_two = arith.cmpi eq, %core_x, %two : index
    %is_active = arith.ori %is_one, %is_two : i1
    scf.if %is_active {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @cycle_vertex_three()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 53 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %is_two = arith.cmpi eq, %core_x, %two : index
    %is_three = arith.cmpi eq, %core_x, %three : index
    %is_active = arith.ori %is_two, %is_three : i1
    scf.if %is_active {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @cycle_vertex_four()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %dfb = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 54 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %core_x = ttl.core_x : index
    %three = arith.constant 3 : index
    %four = arith.constant 4 : index
    %is_three = arith.cmpi eq, %core_x, %three : index
    %is_four = arith.cmpi eq, %core_x, %four : index
    %is_active = arith.ori %is_three, %is_four : i1
    scf.if %is_active {
      %slot = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      %value = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }
}
