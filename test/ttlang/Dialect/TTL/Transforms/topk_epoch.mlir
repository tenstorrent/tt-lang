// RUN: ttlang-opt %s --ttl-verify-topk-epoch --split-input-file | FileCheck %s

// Summary: ttl-verify-topk-epoch accepts a stage that stays on one
// representation. A fused stage either shares its section with topk_fuse or
// reads and stores fused keys. Defuse and strip consume that representation
// and store plain tiles. An empty movement or store list does not invent a
// mismatch. The same rules apply between tile_regs_acquire and
// tile_regs_release.

// CHECK-LABEL: func.func @plain_stage
// CHECK: ttl.tile_topk_local_sort
func.func @plain_stage() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.dst_section {
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @section_without_topk
// CHECK: ttl.copy_tile
func.func @section_without_topk(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  ttl.dst_section {
    %tok, %copied = ttl.copy_tile %tile[%c0, %c0] into dst[%dst]
        : !ttcore.tile<32x32, bf16> -> !ttl.dst, !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @fused_stage_reads_and_stores_fused_keys
// CHECK: ttl.tile_topk_local_sort
// CHECK: ttl.tile_store
func.func @fused_stage_reads_and_stores_fused_keys() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
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
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @fused_stage_transposes_fused_keys
// CHECK: ttl.tile_transpose
// CHECK: ttl.tile_topk_local_sort
func.func @fused_stage_transposes_fused_keys() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
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
    %transposed = ttl.tile_transpose %source, %source into dst[%dst]
        : (!ttcore.tile<32x32, u32>, !ttcore.tile<32x32, u32>) -> !ttcore.tile<32x32, u32>
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %transposed, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @fused_stage_shares_section_with_fuse
// CHECK: ttl.tile_topk_fuse
// CHECK: ttl.tile_topk_local_sort
// CHECK: ttl.tile_store
func.func @fused_stage_shares_section_with_fuse(%tile: !ttcore.tile<32x32, u32>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.dst_section {
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {fused = true, order = #ttl.topk_order<descending>}
        : (index, i32, i32, i32) -> ()
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// An empty store list makes the fuse store check succeed.
// CHECK-LABEL: func.func @fuse_without_a_store
// CHECK: ttl.tile_topk_fuse
func.func @fuse_without_a_store() {
  %dst = arith.constant 0 : index
  ttl.dst_section {
    ttl.tile_topk_fuse dst[%dst] {order = #ttl.topk_order<descending>} : index
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @defuse_reads_fused_keys_and_stores_plain_tiles
// CHECK: ttl.tile_topk_defuse
// CHECK: ttl.tile_store
func.func @defuse_reads_fused_keys_and_stores_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
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
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
    ttl.yield
  }
  return
}

// -----

// Empty movement and stores make the defuse checks succeed.
// CHECK-LABEL: func.func @defuse_without_movement
// CHECK: ttl.tile_topk_defuse
func.func @defuse_without_movement() {
  %dst = arith.constant 0 : index
  %num_tiles = arith.constant 1 : i32
  ttl.dst_section {
    ttl.tile_topk_defuse dst[%dst] num_tiles = %num_tiles
        {order = #ttl.topk_order<descending>} : (index, i32) -> ()
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @stamp_stores_rank_stamped_tiles
// CHECK: ttl.tile_topk_stamp_local_positions
// CHECK: ttl.tile_store
func.func @stamp_stores_rank_stamped_tiles(%tile: !ttcore.tile<32x32, bf16>) {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %view = ttl.cb_reserve %stamped
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    ttl.tile_topk_stamp_local_positions dst[%dst]
        {order = #ttl.topk_order<descending>} : index
    ttl.tile_store %tile, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @rank_stamped_stage_reads_stamped_tiles
// CHECK: ttl.tile_topk_local_sort
// CHECK-SAME: rank_stamped = true
func.func @rank_stamped_stage_reads_stamped_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
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

// CHECK-LABEL: func.func @strip_reads_stamped_tiles_and_stores_plain_tiles
// CHECK: ttl.tile_topk_strip_rank_tags
// CHECK: ttl.tile_store
func.func @strip_reads_stamped_tiles_and_stores_plain_tiles() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %stamped = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<rank_stamped>}
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
    ttl.tile_topk_strip_rank_tags dst[%dst] : index
    ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
        : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @canonicalize_beside_stable_local_sort
// CHECK: ttl.tile_topk_canonicalize_negzero_values
// CHECK: ttl.tile_topk_local_sort
// CHECK-SAME: stable_sort = true
func.func @canonicalize_beside_stable_local_sort() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.dst_section {
    ttl.tile_topk_canonicalize_negzero_values dst[%dst] : index
    ttl.tile_topk_local_sort dst[%dst] direction = %dir
        end_phase = %end start_phase = %start
        {order = #ttl.topk_order<descending>, stable_sort = true}
        : (index, i32, i32, i32) -> ()
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @uint16_move_inside_a_section
// CHECK: ttl.tile_topk_uint16_move_dest_tile_to_pack_half
func.func @uint16_move_inside_a_section() {
  %dst = arith.constant 0 : index
  ttl.dst_section {
    ttl.tile_topk_uint16_move_dest_tile_to_pack_half dst[%dst] : index
    ttl.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @plain_stage_between_acquire_and_release
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.tile_topk_local_sort
// CHECK: ttl.tile_regs_release
func.func @plain_stage_between_acquire_and_release() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// -----

// CHECK-LABEL: func.func @fused_stage_between_acquire_and_release
// CHECK: ttl.tile_regs_acquire
// CHECK: ttl.copy_tile
// CHECK: ttl.tile_topk_local_sort
// CHECK: ttl.tile_regs_release
func.func @fused_stage_between_acquire_and_release() {
  %dst = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %fused = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.topk_payload = #ttl.topk_payload<fused_keys>}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>
  %waited = ttl.cb_wait %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %attached = ttl.attach_cb %waited, %fused
      : (tensor<1x1x!ttcore.tile<32x32, u32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, u32>, 1>)
        -> tensor<1x1x!ttcore.tile<32x32, u32>>
  %source = tensor.extract %attached[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, u32>>
  %view = ttl.cb_reserve %fused
      : <[1, 1], !ttcore.tile<32x32, u32>, 1> -> tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.tile_regs_acquire
  %tok, %copied = ttl.copy_tile %source[%c0, %c0] into dst[%dst]
      : !ttcore.tile<32x32, u32> -> !ttl.dst, !ttcore.tile<32x32, u32>
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {fused = true, order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_store %copied, %view[%c0, %c0] from dst[%dst]
      : !ttcore.tile<32x32, u32>, tensor<1x1x!ttcore.tile<32x32, u32>>
  ttl.tile_regs_release
  return
}
