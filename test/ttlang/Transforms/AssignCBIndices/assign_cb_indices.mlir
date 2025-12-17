// RUN: ttlang-opt %s -ttl-assign-cb-indices -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @assign_missing
// CHECK: ttl.bind_cb() {buffer_factor = 2 : i64, buffer_index = 0 : i32, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
// CHECK: ttl.bind_cb() {buffer_factor = 2 : i64, buffer_index = 1 : i32, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
module {
  func.func @assign_missing() {
    %cb0 = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    func.return
  }
}
