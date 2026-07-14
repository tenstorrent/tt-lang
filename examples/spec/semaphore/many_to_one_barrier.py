# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding) exists so the file can
# stand on its own and is not copied into the specification.

import math

import torch

import ttl
import ttnn

# spec:begin
node_num = ttl.node(dims=1)
my_barrier = ttl.Semaphore()
node_0_barrier = my_barrier.get_remote((0, 0))
non_0_node_count = grid_size(dims=1) - 1


@ttl.datamovement()
def dm():
    if node_num != 0:

        # do something on non-0 nodes while node 0 waits...

        node_0_barrier.inc(1)
    else:
        my_barrier.wait_eq(non_0_node_count)

        # non-0 nodes are done


# spec:end
