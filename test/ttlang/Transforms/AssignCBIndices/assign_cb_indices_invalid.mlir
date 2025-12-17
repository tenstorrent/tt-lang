// RUN: ttlang-opt %s -ttl-assign-cb-indices -verify-diagnostics

module {
  func.func @conflict_types() {
    %cb0 = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2, buffer_index = 0 : i32} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{buffer_index 0 already bound with type '!ttl.cb<[1, 1], f32, 2>'}}
    %cb1 = ttl.bind_cb() {shape = [1, 1], element_type = f16, buffer_factor = 2, buffer_index = 0 : i32} : !ttl.cb<[1, 1], f16, 2>
    func.return
  }
}
