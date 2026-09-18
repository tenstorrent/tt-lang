// Round-trip verification of ttl.get_dfb_id.
// Verifies that the op parses, prints, and round-trips correctly with the
// expected type abbreviation in the assembly format.

// RUN: ttlang-opt %s | FileCheck %s

// Verify basic round-trip of get_dfb_id on a bound CB.
// CHECK-LABEL: func.func @get_dfb_id_basic
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 2, block_count = 1} : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
// CHECK-NEXT: %[[ID:.*]] = ttl.get_dfb_id %[[CB]] : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
func.func @get_dfb_id_basic() -> i32 attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %id = ttl.get_dfb_id %cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  func.return %id : i32
}
