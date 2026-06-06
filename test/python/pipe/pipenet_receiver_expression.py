# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_CASE=compile %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: env TTLANG_CASE=discovery %python %s
# RUN: not env TTLANG_COMPILE_ONLY=1 TTLANG_CASE=invalid_receiver %python %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

"""Compile-only coverage for indexed PipeNet receiver expressions.

Generated communication patterns often keep related PipeNets in Python
containers. The frontend must resolve those compile-time container lookups for
PipeNet predicates, callbacks, and DFB role scopes. PipeNet discovery must also
traverse nested metadata containers so validation sees every referenced net.
"""

import os

import pytest
import torch
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


class BFloat16Tensor:
    dtype = torch.bfloat16


PIPE_NET_GROUPS = {
    "tree": (
        ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))]),
        ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(2, 0))]),
    )
}
PIPE_NET_LIST = [ttl.PipeNet([ttl.Pipe(src=(5, 0), dst=(4, 0))])]
PIPE_NET_BY_TUPLE_KEY = {
    ("mesh", 0): ttl.PipeNet([ttl.Pipe(src=(7, 0), dst=(6, 0))])
}
RUNTIME_PIPE_NETS = [
    ttl.PipeNet([ttl.Pipe(src=(9, 0), dst=(8, 0))]),
    ttl.PipeNet([ttl.Pipe(src=(11, 0), dst=(10, 0))]),
]


def _assert_nested_discovery_helpers():
    from ttl.ttl_api import _iter_pipe_nets_in_value
    from ttl.sim.pipe import (
        Pipe as SimPipe,
        PipeNet as SimPipeNet,
        discover_pipe_nets_from_closures,
    )

    large_pipe_net_list = [
        ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))]) for _ in range(32)
    ]
    nested_ttl_container = {
        PIPE_NET_GROUPS["tree"][0]: [PIPE_NET_LIST[0]],
        "dict": {"tuple_key": PIPE_NET_BY_TUPLE_KEY[("mesh", 0)]},
        "large": large_pipe_net_list,
    }
    discovered_ttl_nets = list(_iter_pipe_nets_in_value(nested_ttl_container, set()))
    assert {id(net) for net in discovered_ttl_nets} == {
        id(PIPE_NET_GROUPS["tree"][0]),
        id(PIPE_NET_LIST[0]),
        id(PIPE_NET_BY_TUPLE_KEY[("mesh", 0)]),
        *(id(net) for net in large_pipe_net_list),
    }

    sim_net_a = SimPipeNet([SimPipe(src=(0, 0), dst=(1, 0))])
    sim_net_b = SimPipeNet([SimPipe(src=(2, 0), dst=(3, 0))])
    sim_container = {"group": [sim_net_a, {"nested": sim_net_b}]}

    def captures_nested_sim_container():
        return sim_container

    discovered_sim_nets = discover_pipe_nets_from_closures(
        captures_nested_sim_container
    )
    assert {id(net) for net in discovered_sim_nets} == {id(sim_net_a), id(sim_net_b)}


@ttl.operation(grid=(4, 1))
def compile_pipenet_receiver_expression():
    template = BFloat16Tensor()
    send_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(template, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if PIPE_NET_GROUPS["tree"][0].is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass
        if PIPE_NET_GROUPS["tree"][1].is_dst():
            with recv_dfb.wait() as _recv_blk:
                pass
        if {"list": PIPE_NET_LIST}["list"][0].is_active():
            with recv_dfb.wait() as _recv_blk:
                pass
        if PIPE_NET_BY_TUPLE_KEY[("mesh", 0)].is_active():
            with recv_dfb.wait() as _recv_blk:
                pass

    @ttl.datamovement()
    def send_dm():
        if PIPE_NET_GROUPS["tree"][0].is_src():
            with send_dfb.reserve() as send_blk:
                PIPE_NET_GROUPS["tree"][0].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if PIPE_NET_GROUPS["tree"][1].is_src():
            with send_dfb.reserve() as send_blk:
                PIPE_NET_GROUPS["tree"][1].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if [PIPE_NET_LIST[0]][0].is_src():
            with send_dfb.reserve() as send_blk:
                [PIPE_NET_LIST[0]][0].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )
        if {"tuple_key": PIPE_NET_BY_TUPLE_KEY[("mesh", 0)]}["tuple_key"].is_src():
            with send_dfb.reserve() as send_blk:
                {"tuple_key": PIPE_NET_BY_TUPLE_KEY[("mesh", 0)]}["tuple_key"].if_src(
                    lambda pipe: ttl.copy(send_blk, pipe).wait()
                )

    @ttl.datamovement()
    def recv_dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        PIPE_NET_GROUPS["tree"][0].if_dst(recv)
        PIPE_NET_GROUPS["tree"][1].if_dst(recv)
        PIPE_NET_LIST[0].if_dst(recv)
        PIPE_NET_BY_TUPLE_KEY[("mesh", 0)].if_dst(recv)


@ttl.operation(grid=(12, 1))
def compile_invalid_dynamic_receiver():
    @ttl.datamovement()
    def dm():
        x, _ = ttl.node(dims=2)
        if RUNTIME_PIPE_NETS[x].is_src():
            pass


def main():
    test_case = os.environ.get("TTLANG_CASE", "compile")
    if test_case == "compile":
        compile_pipenet_receiver_expression()
    elif test_case == "discovery":
        _assert_nested_discovery_helpers()
    elif test_case == "invalid_receiver":
        compile_invalid_dynamic_receiver()
    else:
        raise ValueError(f"unknown TTLANG_CASE: {test_case}")


if __name__ == "__main__":
    main()


# CHECK-INITIAL-LABEL: func.func @compute()
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.is_dst {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.is_active {pipe_net_id = 2 : i64}
# CHECK-INITIAL: ttl.is_active {pipe_net_id = 3 : i64}
# CHECK-INITIAL-LABEL: func.func @send_dm()
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 0 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 0
# CHECK-INITIAL: ttl.if_src
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 1 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [1], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 1
# CHECK-INITIAL: ttl.if_src
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 2 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [2], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 2
# CHECK-INITIAL: ttl.if_src
# CHECK-INITIAL: ttl.is_src {pipe_net_id = 3 : i64}
# CHECK-INITIAL: ttl.pipenet_scope attributes {ttl.pipe_net_ids = [3], ttl.pipe_net_roles = [0]}
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 3
# CHECK-INITIAL: ttl.if_src
# CHECK-INITIAL-LABEL: func.func @recv_dm()
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 0
# CHECK-INITIAL: ttl.if_dst
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 1
# CHECK-INITIAL: ttl.if_dst
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 2
# CHECK-INITIAL: ttl.if_dst
# CHECK-INITIAL: ttl.create_pipe
# CHECK-INITIAL-SAME: net 3
# CHECK-INITIAL: ttl.if_dst

# CHECK-INVALID: PipeNet.is_src() receiver must be a compile-time PipeNet expression
