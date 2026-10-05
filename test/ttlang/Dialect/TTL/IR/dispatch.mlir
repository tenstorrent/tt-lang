// RUN: ttlang-opt %s --canonicalize --ttl-resolve-static-dispatch | FileCheck %s

// Verify that static dispatch keeps target declarations and resolves the
// dispatcher function in place without creating a manifest.

module {
  ttl.dispatch.target @kda : (i32, i32) -> () {
    argument_names = ["input", "handoff"],
    operation_identity = "kda-operation"
  }
  ttl.dispatch.target @moe : (i32, i32) -> () {
    argument_names = ["handoff", "output"],
    operation_identity = "moe-operation"
  }

  func.func @controller(%input: i32, %handoff: i32, %output: i32)
      attributes {ttl.dispatcher} {
    %enabled = arith.constant true
    ttl.dispatch.invoke @kda(%input, %handoff : i32, i32)
    scf.if %enabled {
      ttl.dispatch.invoke @moe(%handoff, %output : i32, i32)
    }
    func.return
  }
}

// CHECK: ttl.dispatch.target @kda : (i32, i32) -> ()
// CHECK-SAME: argument_names = ["input", "handoff"]
// CHECK-SAME: operation_identity = "kda-operation"
// CHECK: ttl.dispatch.target @moe : (i32, i32) -> ()
// CHECK-LABEL: func.func @controller
// CHECK-SAME: attributes {ttl.dispatch.resolved, ttl.dispatcher}
// CHECK-NEXT: ttl.dispatch.invoke @kda(%arg0, %arg1 : i32, i32)
// CHECK-NEXT: ttl.dispatch.invoke @moe(%arg1, %arg2 : i32, i32)
// CHECK-NEXT: return
// CHECK-NOT: manifest
