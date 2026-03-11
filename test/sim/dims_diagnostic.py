#!/usr/bin/env python3
"""Diagnostic: compare TTL sim vs compiler vs PyTorch dims semantics.

Run with:
  cd tt-lang && source build/env/activate
  PYTHONPATH="build/python_packages:python:$PYTHONPATH" python3 test/sim/dims_diagnostic.py
"""

import io
import os
import sys
import torch

# --- Sim imports ---
from sim.dfb import Block
from sim.ttnnsim import Tensor
from sim.math import reduce_sum as sim_reduce_sum, broadcast as sim_broadcast

# --- Compiler imports ---
os.environ["TTLANG_COMPILE_ONLY"] = "1"
import ttnn
import ttl


def make_block(tiles, shape):
    return Block.from_list([Tensor(t) for t in tiles], shape=shape)

def vals(block):
    return [t.to_torch()[0, 0].item() for t in block.to_list()]

def make_ttnn(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def compile_quiet(fn, *args):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        fn(*args)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# --- Compiler kernels (PyTorch semantics) ---

@ttl.kernel(grid=(1, 1))
def compiler_reduce_dims0(inp, scaler, out):
    # dims=[0] collapses rows: (2,2) -> (1,2)
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), buffer_factor=2)
    @ttl.compute()
    def c():
        with inp_dfb.wait() as x, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(x, s, dims=[0]))
    @ttl.datamovement()
    def r():
        with inp_dfb.reserve() as b:
            tx = ttl.copy(inp[0:2, 0:2], b); tx.wait()
        with scaler_dfb.reserve() as b:
            tx = ttl.copy(scaler[0, 0], b); tx.wait()
    @ttl.datamovement()
    def w():
        with out_dfb.wait() as b:
            tx = ttl.copy(b, out[0:1, 0:2]); tx.wait()


@ttl.kernel(grid=(1, 1))
def compiler_reduce_dims1(inp, scaler, out):
    # dims=[1] collapses columns: (2,2) -> (2,1)
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), buffer_factor=2)
    @ttl.compute()
    def c():
        with inp_dfb.wait() as x, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(x, s, dims=[1]))
    @ttl.datamovement()
    def r():
        with inp_dfb.reserve() as b:
            tx = ttl.copy(inp[0:2, 0:2], b); tx.wait()
        with scaler_dfb.reserve() as b:
            tx = ttl.copy(scaler[0, 0], b); tx.wait()
    @ttl.datamovement()
    def w():
        with out_dfb.wait() as b:
            tx = ttl.copy(b, out[0:2, 0:1]); tx.wait()


@ttl.kernel(grid=(1, 1))
def compiler_bcast_dims0(inp, out):
    # dims=[0] expands rows: input (1,2) -> (2,2)
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)
    @ttl.compute()
    def c():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.broadcast(x, dims=[0]))
    @ttl.datamovement()
    def r():
        with inp_dfb.reserve() as b:
            tx = ttl.copy(inp[0:1, 0:2], b); tx.wait()
    @ttl.datamovement()
    def w():
        with out_dfb.wait() as b:
            tx = ttl.copy(b, out[0:2, 0:2]); tx.wait()


@ttl.kernel(grid=(1, 1))
def compiler_bcast_dims1(inp, out):
    # dims=[1] expands columns: input (2,1) -> (2,2)
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)
    @ttl.compute()
    def c():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.broadcast(x, dims=[1]))
    @ttl.datamovement()
    def r():
        with inp_dfb.reserve() as b:
            tx = ttl.copy(inp[0:2, 0:1], b); tx.wait()
    @ttl.datamovement()
    def w():
        with out_dfb.wait() as b:
            tx = ttl.copy(b, out[0:2, 0:2]); tx.wait()


# --- Sim setup ---
scaler = make_block([torch.ones(1, 1)], shape=(1, 1))
t = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
inp = make_block([
    torch.full((1, 1), 1.0),
    torch.full((1, 1), 2.0),
    torch.full((1, 1), 3.0),
    torch.full((1, 1), 4.0),
], shape=(2, 2))

# --- Compiler setup ---
device = ttnn.open_device(device_id=0)
c_inp = make_ttnn(torch.ones(64, 64, dtype=torch.bfloat16))
c_scaler = make_ttnn(torch.ones(32, 32, dtype=torch.bfloat16))

# --- Output (all should match PyTorch) ---

print("=" * 70)
print("REDUCE (PyTorch semantics: dims = dimensions to collapse)")
print("=" * 70)
print("  input (2,2): [[1, 2], [3, 4]]")

print()
print("  --- dims=[0]: collapse rows -> (1,2) ---")
r = sim_reduce_sum(inp, scaler, dims=[0])
p = t.sum(dim=0, keepdim=True)
print(f"  torch.sum(dim=0): shape={tuple(p.shape)}  vals={p.squeeze(0).tolist()}")
print(f"  sim:              shape={r._shape}  vals={vals(r)}")
try:
    c_out = make_ttnn(torch.zeros(32, 64, dtype=torch.bfloat16))
    compile_quiet(compiler_reduce_dims0, c_inp, c_scaler, c_out)
    print("  compiler:         OK (out_dfb=(1,2))")
except Exception as e:
    print(f"  compiler:         ERROR: {e}")

print()
print("  --- dims=[1]: collapse cols -> (2,1) ---")
r = sim_reduce_sum(inp, scaler, dims=[1])
p = t.sum(dim=1, keepdim=True)
print(f"  torch.sum(dim=1): shape={tuple(p.shape)}  vals={p.squeeze(1).tolist()}")
print(f"  sim:              shape={r._shape}  vals={vals(r)}")
try:
    c_out = make_ttnn(torch.zeros(64, 32, dtype=torch.bfloat16))
    compile_quiet(compiler_reduce_dims1, c_inp, c_scaler, c_out)
    print("  compiler:         OK (out_dfb=(2,1))")
except Exception as e:
    print(f"  compiler:         ERROR: {e}")

print()
print("=" * 70)
print("BROADCAST (PyTorch semantics: dims = dimensions with size 1 to expand)")
print("=" * 70)

print()
print("  --- dims=[0]: input (1,2) -> (2,2), expand rows ---")
row_inp = make_block([torch.full((1, 1), 5.0), torch.full((1, 1), 7.0)], shape=(1, 2))
pt = torch.tensor([[5.0, 7.0]])
try:
    b = sim_broadcast(row_inp, dims=[0])
    print(f"  torch expand:     shape={tuple(pt.expand(2,2).shape)}  vals={pt.expand(2,2).flatten().tolist()}")
    print(f"  sim:              shape={b._shape}  vals={vals(b)}")
except Exception as e:
    print(f"  torch expand:     shape={tuple(pt.expand(2,2).shape)}  vals={pt.expand(2,2).flatten().tolist()}")
    print(f"  sim:              ERROR: {e}")
try:
    c_in = make_ttnn(torch.ones(32, 64, dtype=torch.bfloat16))
    c_out = make_ttnn(torch.zeros(64, 64, dtype=torch.bfloat16))
    compile_quiet(compiler_bcast_dims0, c_in, c_out)
    print("  compiler:         OK (inp_dfb=(1,2) out_dfb=(2,2))")
except Exception as e:
    print(f"  compiler:         ERROR: {e}")

print()
print("  --- dims=[1]: input (2,1) -> (2,2), expand cols ---")
col_inp = make_block([torch.full((1, 1), 5.0), torch.full((1, 1), 7.0)], shape=(2, 1))
pt = torch.tensor([[5.0], [7.0]])
try:
    b = sim_broadcast(col_inp, dims=[1])
    print(f"  torch expand:     shape={tuple(pt.expand(2,2).shape)}  vals={pt.expand(2,2).flatten().tolist()}")
    print(f"  sim:              shape={b._shape}  vals={vals(b)}")
except Exception as e:
    print(f"  torch expand:     shape={tuple(pt.expand(2,2).shape)}  vals={pt.expand(2,2).flatten().tolist()}")
    print(f"  sim:              ERROR: {e}")
try:
    c_in = make_ttnn(torch.ones(64, 32, dtype=torch.bfloat16))
    c_out = make_ttnn(torch.zeros(64, 64, dtype=torch.bfloat16))
    compile_quiet(compiler_bcast_dims1, c_in, c_out)
    print("  compiler:         OK (inp_dfb=(2,1) out_dfb=(2,2))")
except Exception as e:
    print(f"  compiler:         ERROR: {e}")

ttnn.close_device(device)
