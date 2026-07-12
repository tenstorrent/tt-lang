# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Graph-based PipeNets require device-aware lowering.

This test pins the frontend boundary: `PipeNet(graph=...)` is a valid object
model, but existing local `ttl.create_pipe` lowering must not consume it.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl  # noqa: E402

domain = ttl.DeviceDomain((1, 2))
graph = ttl.TransferGraph.edges(domain, edges=[((0, 0), (0, 1))])
net = ttl.PipeNet(graph=graph)


# CHECK: error: graph-based PipeNet `net` requires device-aware pipe lowering
@ttl.operation(grid=(1, 1), device_domain=domain)
def invalid_graph_pipenet_lowering():
    @ttl.datamovement()
    def dm_read():
        def send(pipe):
            pass

        net.if_src(send)


invalid_graph_pipenet_lowering()
