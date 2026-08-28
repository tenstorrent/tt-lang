// Verifies that ttl-insert-copy-wait rejects receive lifetimes that cannot be
// completed for every dynamic copy execution.

// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-insert-copy-wait))' --verify-diagnostics

// A final loop result represents only the last dynamic request, so wait_any
// cannot provide cleanup for requests created by earlier iterations.
module {
  func.func @final_loop_request_wait_any() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %start = arith.constant 0 : index
    %loop_landing = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %static_landing = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %loop_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %static_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %static_dst = ttl.cb_reserve %static_landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %static_request = ttl.copy %static_pipe, %static_dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    cf.br ^loop(%c0 : index)
  ^loop(%index: index):
    %loop_dst = ttl.cb_reserve %loop_landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @below {{cannot place an implicit wait after every wait_any observation}}
    %request = ttl.copy %loop_pipe, %loop_dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    %next_index = arith.addi %index, %c1 : index
    %repeat = arith.cmpi ult, %next_index, %c2 : index
    cf.cond_br %repeat, ^loop(%next_index : index),
        ^exit(%request : !ttl.receive_request)
  ^exit(%final_request: !ttl.receive_request):
    %ready = ttl.wait_any %final_request, %static_request start %start
        : (!ttl.receive_request, !ttl.receive_request, index)
        -> !ttl.ready_receive
    func.return
  }
}
