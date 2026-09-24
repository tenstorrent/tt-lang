// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s

// Summary: Verifies computed receiver addresses after static pipe transfer
// expansion, where a receiver post may execute in a narrower control context
// than the push that publishes its DFB block.

// The receiver reserves and pushes one full DFB every iteration but posts only
// when the payload arrives over the pipe. Each push finalizes its own reserve
// and returns the write pointer to the same slot, so the sender still computes
// the address.

// CHECK-LABEL: func.func @conditional_post_full_push_computes
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @conditional_post_full_push_computes() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %reserved = ttl.cb_reserve %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x2x!ttcore.tile<32x32, f32>>
        %received = arith.cmpi ne, %iteration, %zero : index
        scf.if %received {
          %post = ttl.copy %pipe, %reserved
              : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
                 tensor<1x2x!ttcore.tile<32x32, f32>>)
              -> !ttl.receive_request
          ttl.wait %post : !ttl.receive_request
        }
        ttl.cb_push %dst {num_tiles = 2 : i64}
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      scf.for %iteration = %zero to %two step %one {
        %sent = arith.cmpi ne, %iteration, %zero : index
        scf.if %sent {
          %send = ttl.copy %src, %pipe
              : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>,
                 !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
    }
    func.return
  }
}
