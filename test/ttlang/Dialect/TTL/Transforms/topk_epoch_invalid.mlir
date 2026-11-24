// RUN: ttlang-opt %s --ttl-verify-topk-epoch --split-input-file --verify-diagnostics

// Summary: ttl-verify-topk-epoch rejects a stage or helper outside a
// destination-register section and a section whose packed representation does
// not stay on one edge. The sort order is an epoch invariant: a section with
// two orders, or one that reads or stores a packed buffer of another order,
// is rejected.

func.func @fused_stage_without_fused_keys() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.dst_section {
    // expected-error @below {{a fused TopK stage must read fused keys or share its section with topk_fuse}}
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @fuse_and_defuse_share_a_section() {
  %dst = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  ttl.dst_section {
    // expected-error @below {{topk_fuse and topk_defuse cannot share a destination-register section}}
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @stage_outside_a_section() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{TopK operation must be inside a destination-register section}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @helper_outside_a_section() {
  %dst = arith.constant 0 : index
  // expected-error @below {{TopK operation must be inside a destination-register section}}
  ttl.tile_topk_uint16_move_dest_tile_to_pack_half dst[%dst] : index
  return
}

// -----

func.func @fused_stage_in_acquire_without_keys() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.tile_regs_acquire
  // expected-error @below {{a fused TopK stage must read fused keys or share its section with topk_fuse}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {fused = true, order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// -----

func.func @stages_disagree() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  ttl.dst_section {
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    // expected-error @below {{TopK stages in one section must agree on fused, rank_stamped, and stable_sort}}
    ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
        {order = #ttl.topk_order<descending>} : (index, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @stamp_and_strip_share_a_section() {
  %dst = arith.constant 0 : index
  ttl.dst_section {
    // expected-error @below {{topk_stamp_local_positions and topk_strip_rank_tags cannot share a destination-register section}}
    ttl.tile_topk_stamp_local_positions dst[%dst]
        {order = #ttl.topk_order<descending>} : index
    ttl.tile_topk_strip_rank_tags dst[%dst] : index
    ttl.yield
  }
  return
}

// -----

func.func @fused_and_rank_stamped_helpers_share_a_section() {
  %dst = arith.constant 0 : index
  ttl.dst_section {
    // expected-error @below {{fused and rank-stamped helpers cannot share a destination-register section}}
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    ttl.tile_topk_stamp_local_positions dst[%dst]
        {order = #ttl.topk_order<descending>} : index
    ttl.yield
  }
  return
}

// -----

func.func @canonicalize_without_stable_local_sort() {
  %dst = arith.constant 0 : index
  ttl.dst_section {
    // expected-error @below {{canonicalize_negzero requires a stable local sort in the same section}}
    ttl.tile_topk_canonicalize_negzero_values dst[%dst] : index
    ttl.yield
  }
  return
}

// -----

func.func @fused_stage_with_defuse() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %num_tiles = arith.constant 1 : i32
  ttl.dst_section {
    // expected-error @below {{a fused TopK stage cannot share its section with topk_defuse}}
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @rank_stamped_stage_with_strip() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.dst_section {
    // expected-error @below {{a rank-stamped TopK stage cannot share its section with topk_strip_rank_tags}}
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>, rank_stamped = true}
        : (index, i32, i32, i32) -> ()
    ttl.tile_topk_strip_rank_tags dst[%dst] : index
    ttl.yield
  }
  return
}

// -----

func.func @rank_stamped_stage_without_stamped_tiles() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.dst_section {
    // expected-error @below {{a rank-stamped TopK stage must read rank-stamped tiles or share its section with topk_stamp_local_positions}}
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>, rank_stamped = true}
        : (index, i32, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @fuse_stores_plain_tiles(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    // expected-error @below {{topk_fuse must store fused_keys tiles}}
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @defuse_reads_plain_tiles(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %waited, %plain
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    // expected-error @below {{topk_defuse must read fused_keys tiles}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @stamp_stores_plain_tiles(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    ttl.tile_topk_stamp_local_positions dst[%dst]
        {order = #ttl.topk_order<descending>} : index
    // expected-error @below {{topk_stamp_local_positions must store rank_stamped tiles}}
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @strip_reads_plain_tiles(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %waited, %plain
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    // expected-error @below {{topk_strip_rank_tags must read rank_stamped tiles}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_topk_strip_rank_tags dst[%dst] : index
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @fused_stage_reads_plain_tiles(%keys: tensor<1x2x!ttcore.tile<32x32, u32>>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u32>, 1>
  %fused = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %plain
      : <[1, 2], !ttcore.tile<32x32, u32>, 1> -> tensor<1x2x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %plain
      : (tensor<1x2x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x2x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %fused
      : <[1, 2], !ttcore.tile<32x32, u32>, 1> -> tensor<1x2x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    // expected-error @below {{a fused TopK stage must read fused_keys tiles}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x2x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

func.func @fused_stage_stores_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %plain = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %fused
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    // expected-error @below {{a fused TopK stage must store fused_keys tiles}}
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

func.func @defuse_stores_fused_keys() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %fused
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    // expected-error @below {{topk_defuse must store plain tiles}}
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

func.func @defuse_transposes_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %plain
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %plain
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    // expected-error @below {{topk_defuse must read fused_keys tiles}}
    %transposed = ttl.tile_transpose %source, %source into dst[%dst]
        : (!ttcore.tile<32x32, u32>, !ttcore.tile<32x32, u32>) -> !ttcore.tile<32x32, u32>
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.yield
  }
  return
}

// -----

func.func @rank_stamped_stage_reads_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %stamped = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %waited, %plain
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %stamped
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    // expected-error @below {{a rank-stamped TopK stage must read rank_stamped tiles}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>, rank_stamped = true}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @rank_stamped_stage_stores_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %plain = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %stamped
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %waited, %stamped
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>, rank_stamped = true}
        : (index, i32, i32, i32) -> ()
    // expected-error @below {{a rank-stamped TopK stage must store rank_stamped tiles}}
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

func.func @strip_stores_rank_stamped_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %stamped
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached = ttl.attach_cb %waited, %stamped
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %view = ttl.cb_reserve %stamped
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.tile_topk_strip_rank_tags dst[%dst] : index
    // expected-error @below {{topk_strip_rank_tags must store plain tiles}}
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

// The fuse polarity encodes the keys; a stage of the other order in the same
// section would sort them wrongly.
func.func @fuse_and_stage_disagree_on_order(%tile: !ttcore.tile<32x32, u32>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 1 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    // expected-error @below {{TopK operations in one section must agree on order}}
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<ascending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// Plain stages are held to the same rule: one section, one order.
func.func @plain_stages_disagree_on_order() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  ttl.dst_section {
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    // expected-error @below {{TopK operations in one section must agree on order}}
    ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
        {order = #ttl.topk_order<ascending>}
        : (index, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

// A fused stage may only read keys packed with its own order.
func.func @fused_stage_reads_keys_of_other_order() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %fused
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    // expected-error @below {{reads a fused_keys buffer whose order is not ascending}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
    ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
        {direction = true, fused = true, order = #ttl.topk_order<ascending>}
        : (index, i32, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// Keys are packed only into a buffer that records the same order.
func.func @fuse_stores_keys_into_buffer_of_other_order(%tile: !ttcore.tile<32x32, u32>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 1 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<ascending>} : index
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<ascending>}
        : (index, i32, i32, i32) -> ()
    // expected-error @below {{stores into a fused_keys buffer whose order is not ascending}}
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// Defuse must use the polarity the keys were packed with.
func.func @defuse_reads_keys_of_other_order(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<ascending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %plain = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %waited = ttl.cb_wait %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %fused
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %plain
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    // expected-error @below {{reads a fused_keys buffer whose order is not descending}}
    %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

// Rank tags encode the polarity as well.
func.func @stamp_stores_into_buffer_of_other_order(%tile: !ttcore.tile<32x32, u32>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<ascending>, ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %view = ttl.cb_reserve %stamped
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    ttl.tile_topk_stamp_local_positions dst[%dst]
        {order = #ttl.topk_order<descending>, tag_bits = 8 : i32} : index
    // expected-error @below {{stores into a rank_stamped buffer whose order is not descending}}
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// A packed buffer without a recorded order cannot be checked.
func.func @fused_keys_buffer_without_order() {
  // expected-error @below {{a fused_keys buffer must carry ttl.topk_order}}
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  return
}

// -----

func.func @rank_stamped_buffer_without_order() {
  // expected-error @below {{a rank_stamped buffer must carry ttl.topk_order}}
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  return
}

// -----

// A plain buffer has no packed polarity to record.
func.func @plain_buffer_with_order() {
  // expected-error @below {{ttl.topk_order applies only to a fused_keys or rank_stamped buffer}}
  %plain = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_order = #ttl.topk_order<descending>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}
