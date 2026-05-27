// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Negative verifier tests for PipeNet foreach ops.
//
// Note: the empty-records case (`pipes []`) is rejected at parse time by the
// typed ArrayRefParameter, before the op verifier runs; the "requires at least
// one pipe record" check protects programmatic construction.

func.func @mixed_unicast_multicast_records() {
  // expected-error @below {{all pipe records must be either unicast or multicast}}
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isMulticast = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}

// -----

func.func @wrong_region_argument_type() {
  // expected-error @below {{body argument must have type '!ttl.selected_pipe_dst'}}
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    ttl.yield
  }
  func.return
}
