# Plan: `run_*` Wrapper Functions for Metal Examples

## Goal

Add `run_*` wrapper functions to each metal matmul example that separate
kernel execution from test boilerplate. This lets `compare_matmul_perf.py`
import and call each kernel directly.

## Design

- Each `run_*` wrapper is a pure kernel executor: takes `device`,
  pre-allocated tensors, and all blocking/grid params as arguments, returns
  the output tensor. No blocking computation, no printing, no metadata.
- Blocking params are computed by callers using existing utilities
  (`ttnn.split_work_to_cores`, `get_large_matmul_params`) or manual values.
- The existing `test_*` function computes blocking, calls the wrapper, and
  checks correctness.

## Wrapper Signatures

### `single_node_matmul/metal/single_node_matmul.py`

No blocking (single core). Wrapper just takes device + tensors.

```python
def run_single_node_matmul(device, a_tensor, b_tensor, output_tensor):
```

### `multinode_matmul/metal/multinode_matmul.py`

Caller uses `ttnn.split_work_to_cores` to get grid/work params.

```python
def run_multinode_matmul(device, a_tensor, b_tensor, output_tensor,
                         all_nodes, node_group_1, node_group_2,
                         work_per_node1, work_per_node2):
```

### `multinode_reuse_matmul/metal/multinode_reuse_matmul.py`

Caller uses `get_large_matmul_params` from `utils.block_allocation`.

```python
def run_multinode_reuse_matmul(device, a_tensor, b_tensor, output_tensor,
                               K_block_size, per_node_M, per_node_N,
                               out_subblock_h, out_subblock_w):
```

### `1d_mcast_matmul/metal/1d_matmul_metal.py`

Blocking externally specified via `@pytest.mark.parametrize` or manual values.

```python
def run_1d_matmul(device, a_tensor, b_tensor, output_tensor,
                  block_m, block_n, block_k, n_blocks_per_node,
                  subblock_h, subblock_w):
```

## `compare_matmul_perf.py` Changes

| Aspect           | Old                           | New                          |
|------------------|-------------------------------|------------------------------|
| Import paths     | `singlecore_matmul.*`         | `single_node_matmul.*`       |
|                  | `multicore_matmul.*`          | `multinode_matmul.*`         |
|                  | `multicore_reuse_matmul.*`    | `multinode_reuse_matmul.*`   |
| Wrapper names    | `run_singlecore_matmul`       | `run_single_node_matmul`     |
|                  | `run_multicore_matmul`        | `run_multinode_matmul`       |
|                  | `run_multicore_reuse_matmul`  | `run_multinode_reuse_matmul` |
| CLI keys         | `singlecore`, `multicore`, .. | `single_node`, `multinode`,  |
|                  | `multicore_reuse`             | `multinode_reuse`            |

Blocking logic in comparison functions stays external (already was).
