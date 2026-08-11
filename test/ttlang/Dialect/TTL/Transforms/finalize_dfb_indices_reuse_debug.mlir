// Tests default-mode debug reporting when three logical DFBs use two physical
// indices.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' 2>&1 | FileCheck %s --check-prefix=NO-REPORT

// REPORT: DFB allocation liveness report
// REPORT: DFB logical_id=0 bounded=1 compiler_created=0
// REPORT: access 0 effect=reserve tiles=1 sequence=0 domain={(0,0)} operation=ttl.cb_reserve kernel=@producer
// REPORT: node (0,0) quiescence=none domain_assumption=exact may_be_active=1 evidence=none occurrences=[0:1, 1:1, 2:1, 3:1] transactions=[1] write_owner=(0,0):noc0:write read_owner=(0,0):unpack:read
// REPORT: DFB conflict lhs=0 rhs=1 reason=pointer-owner-mismatch node=(0,0)
// REPORT: DFB allocation liveness report end
// REPORT-NEXT: Total DFB count: 2
// REPORT-NEXT: DFB assignment: logical DFB 0 -> physical index 0 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 2 -> physical index 0 (bounded)

// NO-REPORT-NOT: DFB allocation liveness report
// NO-REPORT: func.func @producer
// NO-REPORT-NOT: DFB allocation liveness report

module {
  func.func @producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment_block = ttl.cb_wait %acknowledgment
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %acknowledgment : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }

  func.func @consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_block = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %acknowledgment_block = ttl.cb_reserve %acknowledgment
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %acknowledgment : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_block = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
