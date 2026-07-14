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
all_barrier = my_barrier.get_remote_multicast()


@ttl.datamovement()
def dm():
    if node_num == 0:

        # do something on node 0 while non-0 nodes wait...

        all_barrier.set(1)
    else:
        my_barrier.wait_eq(1)

        # node 0 is done


# spec:end
