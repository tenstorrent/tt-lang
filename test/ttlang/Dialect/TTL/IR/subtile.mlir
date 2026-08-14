// RUN: ttlang-opt %s | FileCheck %s

// Verifies that DFB declarations accept every tile dimension constructible by
// tt-metal, including BFP storage at dimensions unsupported by compute LLKs.

// CHECK-LABEL: func.func @storage_tile_dimensions
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<32x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<16x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<32x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<16x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<4x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<2x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<1x32, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<4x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<2x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<1x16, bf16>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_f8>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_bf8>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_f4>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_bf4>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_f2>
// CHECK-NEXT:  %{{.*}} = ttl.bind_cb{{.*}}!ttcore.tile<8x32, bfp_bf2>
// CHECK-NEXT:  return
func.func @storage_tile_dimensions() {
  %dfb0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %dfb1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 1>
  %dfb2 = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x16, bf16>, 1>
  %dfb3 = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<16x16, bf16>, 1>
  %dfb4 = ttl.bind_cb {cb_index = 4, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
  %dfb5 = ttl.bind_cb {cb_index = 5, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<4x32, bf16>, 1>
  %dfb6 = ttl.bind_cb {cb_index = 6, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<2x32, bf16>, 1>
  %dfb7 = ttl.bind_cb {cb_index = 7, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 1>
  %dfb8 = ttl.bind_cb {cb_index = 8, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x16, bf16>, 1>
  %dfb9 = ttl.bind_cb {cb_index = 9, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<4x16, bf16>, 1>
  %dfb10 = ttl.bind_cb {cb_index = 10, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<2x16, bf16>, 1>
  %dfb11 = ttl.bind_cb {cb_index = 11, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 1>
  %dfb12 = ttl.bind_cb {cb_index = 12, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_f8>, 1>
  %dfb13 = ttl.bind_cb {cb_index = 13, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_bf8>, 1>
  %dfb14 = ttl.bind_cb {cb_index = 14, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_f4>, 1>
  %dfb15 = ttl.bind_cb {cb_index = 15, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_bf4>, 1>
  %dfb16 = ttl.bind_cb {cb_index = 16, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_f2>, 1>
  %dfb17 = ttl.bind_cb {cb_index = 17, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<8x32, bfp_bf2>, 1>
  func.return
}
