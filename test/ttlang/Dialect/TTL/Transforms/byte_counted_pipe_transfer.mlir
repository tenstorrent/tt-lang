// RUN: ttlang-opt %s -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' -debug-only=ttl-pipe-transport-plan 2>&1 >/dev/null | FileCheck %s --check-prefix=PLAN
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=true pipe-capacity-sync=false})' | FileCheck %s --check-prefix=LOWER

// Verify one-page planning and lowering for a byte-counted PipeNet transfer.

// PLAN: PipeTransport: stream 0 transfer 0 src(0, 0) -> dst(1, 0)
// PLAN-NEXT: PipeTransport:   source {{.*}} pages=1 page_bytes=896
// PLAN-NEXT: PipeTransport:   endpoint 0 {{.*}}
// LOWER-LABEL: func.func @byte_counted_pipe_transfer
// LOWER: %[[SIZE:.*]] = arith.constant 896 : i32
// LOWER: ttkernel.noc_async_write {{.*}}, core[{{.*}}, {{.*}}], {{.*}}, %[[SIZE]], noc {{.*}}
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @byte_counted_pipe_transfer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dst_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %pipe {
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %token = ttl.pipe_transfer.post %transfer, %recv {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_dfb {
          byte_count = 896 : i64}
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
