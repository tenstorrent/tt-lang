# Implementation Status

This table records the first TT-Lang release that supports each feature in the
functional simulator and compiler. N/S means not supported.

| Functionality | Simulator | Compiler |
| :---- | :---- | :---- |
| Single-device grid `ttl.grid_size` and `ttl.node` with `dims=2`| 0.1.7 | 0.1.7 |
| Single-device grid `ttl.grid_size` and `ttl.node` with any `dims` | 0.1.7 | N/S |
| `ttl.DeviceDomain`, `ttl.DeviceRef`, and `ttl.TransferGraph` | N/S | 1.0.0 |
| [TT-NN Mesh Devices](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) | 0.1.8 | 0.1.8 |
| [TT-NN L1 Sharded Tensors](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md) | 0.1.8 | 0.1.8 |
| `ttl.make_dataflow_buffer_like` with higher than two-dimensional `shape` | 0.1.7 | 0.1.7 |
| `ttl.make_dataflow_buffer_like` for tiled tensors | 0.1.7 | 0.1.7 |
| `ttl.make_dataflow_buffer_like` for row-major tensors | 0.1.8 | N/S |
| `ttl.make_dfb` | N/S | 1.0.0 |
| `ttl.make_tensor_backed_dfb` | N/S | 1.0.0 |
| `ttl.make_dfb_allocation_group` | N/S | 1.0.0 |
| `ttl.Block.store` | 0.1.7 | 0.1.7 |
| Waited-block replacement with `ttl.Block.store` | 1.0.0 | 1.0.0 |
| External DFB inspection with `ttl.DFBAccess.inspect` | N/S | 1.0.0 |
| Overwriting and accumulation through summation (`+=`) for block expressions | 0.1.7 | 1.0.0 |
| `ttl.copy` and `ttl.CopyTransferHandler` | 0.1.7 | 0.1.7 |
| `ttl.ReceiveRequest`, `ttl.ReadyReceive`, and `ttl.wait_any` | 1.0.0 | 1.0.0 |
| `ttl.GroupTransfer` | 1.0.0 | N/S |
| `ttl.Semaphore` on 2D grid | N/S | N/S |
| `ttl.Semaphore` on 4D grid | N/S | N/S |
| `ttl.PipeNet` and `ttl.Pipe` on single-device grid | 0.1.7 | 1.0.0 |
| Graph-based `ttl.PipeNet` with point-to-point device edges | N/S | 1.0.0 |
| `ttl.signpost` (ignored in simulator) | 0.1.7 | 0.1.7 |
| Debug printing with `print` | 0.1.7 | 0.1.7 |
| Built-in unary math operators: `-`, `abs` | 0.1.7 | 0.1.7 |
| Built-in binary math operators: `+`, `-`, `*`, `/` | 0.1.7 | 0.1.7 |
| Built-in binary math operators: `@` | 0.1.7 | 0.1.8 |
| Built-in binary math operators: `%`, `//`, `^`, | 0.1.7 | N/S |
| `ttl.math` unary math functions:<br>`exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `relu`, `floor`, `recip` | 0.1.7 | 0.1.7 |
| `ttl.math` unary math functions:<br>`sin`, `cos`, `tan`, `asin`, `acos`, `atan` | 0.1.7 | 0.1.8 |
| `ttl.math` unary math functions:<br>`expm1`, `exp2`, `ceil`, `sign`, `gelu`, `silu`, `hardsigmoid`, `square`, `softsign`<br>`signbit`, `frac`, `trunc` | 0.1.7 | 1.0.0 |
| `ttl.math` unary math functions:<br>`logp1`, `atanh`, `asinh`, `acosh`, `selu`, `rsub`, `relu_max`, `relu_min`, `leaky_relu`<br>`elu`, `celu`, `prelu`, `softplus`, `hardtanh`, `round`, `clamp`, `threshold` | 0.1.7 | N/S |
| `ttl.math` binary math functions: `min`, `max` | 0.1.7 | 0.1.7 |
| `ttl.block` mask functions: `mask`, `mask_posinf` | 0.1.7 | N/S |
| `ttl.block.where` | 0.1.7 | N/S |
| `ttl.block.broadcast` | 0.1.7 | 0.1.7 |
| `ttl.block.fill` | 0.1.7 | 0.1.8 |
| `ttl.math.reduce_max` | 0.1.7 | 0.1.8 |
| `ttl.math.reduce_sum` | 0.1.7 | 0.1.8 |
| `ttl.block.transpose` | 0.1.7 | 0.1.8 |
| `ttl.block` shape manipulation functions: `squeeze`, `unsqueeze` | N/S | N/S |
| `>` for result of `ttl.raw_element_read` | 1.0.0 | 1.0.0 |
| `<` for result of `ttl.raw_element_read` | 1.0.0 | 1.0.0 |
| `==` for result of `ttl.raw_element_read` | N/S | N/S |
| `!=` for result of `ttl.raw_element_read` | N/S | N/S |
| `>=` for result of `ttl.raw_element_read` | N/S | N/S |
| `<=` for result of `ttl.raw_element_read` | N/S | N/S |
| `ttl.read_index` | N/S | 1.0.0 |
