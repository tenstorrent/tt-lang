// Tests deterministic default-mode debug reporting and conflict evidence.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' > %t.no-report.mlir 2> %t.no-report.log
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices > %t.report.mlir 2> %t.report.log
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices > %t.repeat.mlir 2> %t.repeat.log
// RUN: diff %t.no-report.mlir %t.report.mlir
// RUN: diff %t.report.mlir %t.repeat.mlir
// RUN: diff %t.report.log %t.repeat.log
// RUN: FileCheck %s --check-prefix=REPORT < %t.report.log
// RUN: FileCheck %s --check-prefix=NO-REPORT --allow-empty < %t.no-report.log

// REPORT: DFB allocation liveness report
// REPORT: DFB logical_id=0 bounded=1 compiler_created=0
// REPORT: access 0 effect=reserve tiles=1 sequence=0 domain={(0,0)} operation=ttl.cb_reserve kernel=@producer
// REPORT: node (0,0) quiescence=none domain_assumption=exact evidence=none occurrences=[0:1, 1:1, 2:1, 3:1] transactions=[1] write_owner=(0,0):noc0:write read_owner=(0,0):unpack:read
// REPORT-SAME: earliest_accesses=[0, 2] terminal_accesses=[3]
// REPORT: DFB conflict lhs=0 rhs=1 reason=pointer-owner-mismatch node=(0,0)
// REPORT: DFB allocation liveness report end
// REPORT-NEXT: Total DFB count: 2
// REPORT-NEXT: DFB assignment: logical DFB 0 -> physical index 0 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 1 -> physical index 1 (bounded)
// REPORT-NEXT: DFB assignment: logical DFB 2 -> physical index 0 (bounded)
// REPORT: DFB conflict {{.*}} reason=storage-mismatch

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

// -----

module {
  func.func @different_tensor_backing()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 1, byte_offset = 0, byte_size = 64>}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %first_waited = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second_waited = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x16, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    return
  }
}
