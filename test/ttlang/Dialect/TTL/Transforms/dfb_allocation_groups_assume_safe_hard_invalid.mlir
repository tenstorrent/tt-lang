// Tests errors that unsafe allocation-group assumptions cannot override.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true unsafe-assume-allocation-groups=true})'

// One physical descriptor cannot reinterpret its page format.

module {
  func.func @incompatible_page_formats()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] has incompatible element types for logical DFBs 0 and 1: !ttcore.tile<32x32, bf16> versus !ttcore.tile<32x32, f32>}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// Unknown launch domains cannot hide incompatible physical storage.

module {
  func.func @unknown_domain_storage_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 8 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<4> members=[8, 9] cannot alias logical DFBs 8 and 9: storage-mismatch}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<4>, dfb_id = 9 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// One physical index cannot refer to different storage on one launch node.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @incompatible_tensor_storage()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<1> members=[2, 3] cannot alias logical DFBs 2 and 3: storage-mismatch}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_producer = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_consumer = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_producer = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_consumer = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Unsafe policy may supply a missing handoff relation, but it cannot make a
// cyclic producer/consumer order executable. Each kernel waits for the DFB
// produced only after the other kernel's wait.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @consume_first_produce_second()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "first", operation = "cyclic_order">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<5> members=[10, 11] has contradictory cursor order involving logical DFB 10 on launch node (0,0)}}
    %first_read = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_write = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @consume_second_produce_first()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "second", operation = "cyclic_order">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 10 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<5>, dfb_id = 11 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_read = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_write = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// One physical DFB index requires one compatible static compute
// configuration even when unsafe runtime handoff is enabled.

module {
  func.func @static_configuration_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 6 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<3>, dfb_id = 7 : index}
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
    // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<3> members=[6, 7] cannot alias logical DFBs 6 and 7: static-configuration-mismatch}}
    %broadcast = ttl.tile_bcast %second_tile, %second_tile 1 : i32
        into dst[%zero]
        : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
          -> !ttcore.tile<32x32, f32>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    return
  }
}

// -----

// A larger physical ring must support every transaction sequence from an
// assumed epoch boundary. Two two-tile transactions cross a three-tile ring.

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @ring_envelope_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      // expected-error @below {{DFB allocation group #ttl.dfb_allocation_group<2> members=[4, 5] physical envelope of 3 tiles makes logical DFB 4 cross the ring boundary on launch node (0,0)}}
      ttl.opaque_call "produce_two" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>, #ttl.dfb_protocol_effect<push, 0, 2>] () {header = "producer.hpp"} : () -> ()
    }
    %second_slot = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }

  func.func @ring_envelope_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 4 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<2>, dfb_id = 5 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "consume_two" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    }
    %second_slot = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
    return
  }
}
