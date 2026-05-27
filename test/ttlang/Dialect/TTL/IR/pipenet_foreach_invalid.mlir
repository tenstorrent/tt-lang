// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Negative verifier tests for PipeNet foreach ops.

func.func @empty_pipe_records() {
  // expected-error @below {{requires at least one pipe record}}
  ttl.pipenet_foreach_src attributes {pipeNetId = 0 : i64, pipes = []} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

func.func @mixed_unicast_multicast_records() {
  // expected-error @below {{all pipe records must be either unicast or multicast}}
  ttl.pipenet_foreach_src attributes {
      pipeNetId = 0 : i64,
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isMulticast = true>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

func.func @wrong_region_argument_type() {
  // expected-error @below {{body argument must have type '!ttl.selected_pipe_dst'}}
  ttl.pipenet_foreach_dst attributes {
      pipeNetId = 0 : i64,
      pipes = [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}
