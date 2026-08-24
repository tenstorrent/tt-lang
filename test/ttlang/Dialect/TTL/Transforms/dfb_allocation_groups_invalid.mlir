// Tests rejected DFB allocation-group contracts.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

// A group cannot override an interference edge from concurrent lifecycles.

module {
  func.func @concurrent_lifecycles()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] cannot alias logical DFBs 0 and 1: concurrent-lifetime}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Cursor safety is recomputed against the largest group capacity. Two
// two-tile transactions wrap safely in the logical two-tile DFB, but the
// second transaction would straddle a three-tile physical envelope.

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @envelope_cursor_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "envelope_cursor">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<6> members=[11, 12] physical envelope of 3 tiles makes logical DFB 11 cross the ring boundary on launch node (0,0)}}
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<6>, dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<6>, dfb_id = 12 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "produce_two" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "envelope_cursor">, <kind = data_movement, identity = "reader", operation = "envelope_cursor">, <kind = data_movement, identity = "writer", operation = "envelope_cursor">]>(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %second_slot = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @envelope_cursor_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "envelope_cursor">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<6>, dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<6>, dfb_id = 12 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "consume_two" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    }
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "envelope_cursor">, <kind = data_movement, identity = "reader", operation = "envelope_cursor">, <kind = data_movement, identity = "writer", operation = "envelope_cursor">]>(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    %second_slot = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @envelope_cursor_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "envelope_cursor">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<6>, dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "envelope_cursor">, <kind = data_movement, identity = "reader", operation = "envelope_cursor">, <kind = data_movement, identity = "writer", operation = "envelope_cursor">]>(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
    return
  }
}

// -----

// A group cannot replace proof that every member reaches reusable protocol
// state.

module {
  func.func @access_completion_not_proven()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<3> members=[5, 6] cannot alias logical DFBs 5 and 6: access-completion-not-proven}}
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Tensor-backed members cannot use one enlarged scratch-style capacity
// envelope because their byte ranges describe exact storage.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @tensor_backed_capacity_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 7 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<4> members=[7, 8] cannot use a static capacity envelope for tensor-backed logical DFBs 7 and 8}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 4}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 8 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 8192>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    return
  }
}

// -----

// One physical DFB index must have one compatible static compute
// configuration even when its lifecycles are sequential.

module {
  func.func @static_configuration_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 9 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %first_wait = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %first_attached = ttl.attach_cb %first_wait, %first
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %first_tile = tensor.extract %first_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    %exponential = ttl.tile_exp %first_tile into dst[%zero]
        : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %second_wait = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %second_attached = ttl.attach_cb %second_wait, %second
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %second_tile = tensor.extract %second_attached[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<5> members=[9, 10] cannot alias logical DFBs 9 and 10: static-configuration-mismatch}}
    %broadcast = ttl.tile_bcast %second_tile, %second_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
          -> !ttcore.tile<32x32, f32>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// A capacity envelope cannot reinterpret the element type.

module {
  func.func @incompatible_element_types()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<1> members=[2, 3] has incompatible element types for logical DFBs 2 and 3: !ttcore.tile<32x32, bf16> versus !ttcore.tile<32x32, f32>}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// Every declaration of one logical DFB must preserve the group identity.

module {
  func.func @partial_group_first()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
  func.func @partial_group_second()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    // expected-error @below {{logical DFB 4 has inconsistent allocation groups across kernel functions: expected #ttl.dfb_allocation_group<2> but found none}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// One physical DFB descriptor cannot reinterpret its page geometry.

module {
  func.func @incompatible_page_geometry()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<7>, dfb_id = 13 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<7> members=[13, 14] has incompatible element types for logical DFBs 13 and 14: !ttcore.tile<32x32, bf16> versus !ttcore.tile<1x32, bf16>}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<7>, dfb_id = 14 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A static group cannot change the hardware thread that owns either ring
// pointer.

module {
  func.func @pointer_owner_noc0()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<9>, dfb_id = 17 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
  func.func @pointer_owner_noc1()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<9> members=[17, 18] cannot alias logical DFBs 17 and 18: pointer-owner-mismatch}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<9>, dfb_id = 18 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Unknown launch-node membership cannot establish a safe handoff to an exact
// lifecycle.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @unknown_launch_domain(%runtime_offset: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<10>, dfb_id = 19 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<10> members=[19, 20] cannot alias logical DFBs 19 and 20: unknown-launch-node-domain}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<10>, dfb_id = 20 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %runtime_sum = arith.addi %core_x, %runtime_offset : index
    %runtime_condition = arith.cmpi eq, %runtime_sum, %zero : index
    scf.if %runtime_condition {
      ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// Distinct same-core tensor ranges require different physical storage.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @conflicting_tensor_ranges()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<11>, dfb_id = 21 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<11> members=[21, 22] cannot alias logical DFBs 21 and 22: storage-mismatch}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<11>, dfb_id = 22 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

// A reset can canonicalize queue state but cannot reconfigure one physical
// index from tensor-backed storage to scratch storage.

module attributes {ttl.launch_grid = array<i64: 1, 1>, ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @tensor_to_scratch_transition()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "tensor_to_scratch">,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %tensor_backed = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<12>, dfb_id = 23 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<12> members=[23, 24] cannot alias logical DFBs 23 and 24: storage-mismatch}}
    %scratch = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<12>, dfb_id = 24 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "tensor_use" dfb_dependencies(%tensor_backed : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_to_scratch">, <kind = data_movement, identity = "reader", operation = "tensor_to_scratch">, <kind = data_movement, identity = "writer", operation = "tensor_to_scratch">]>
    ttl.opaque_call "scratch_use" dfb_dependencies(%scratch : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @tensor_to_scratch_reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "tensor_to_scratch">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_to_scratch">, <kind = data_movement, identity = "reader", operation = "tensor_to_scratch">, <kind = data_movement, identity = "writer", operation = "tensor_to_scratch">]>
    return
  }

  func.func @tensor_to_scratch_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "tensor_to_scratch">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "tensor_to_scratch">, <kind = data_movement, identity = "reader", operation = "tensor_to_scratch">, <kind = data_movement, identity = "writer", operation = "tensor_to_scratch">]>
    return
  }
}
