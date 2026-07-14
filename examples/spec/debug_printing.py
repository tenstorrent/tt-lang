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
@ttl.datamovement()
def matmul_read():
    # Print first two pages of c

    print("c: ", c, num_pages=2)

    # Print first page of a and b

    print("a: ", a)
    print("b: ", b)

    for i_tile in range(I_TILES):
        for m_tile in range(M_TILES):
            for n_tile in range(N_TILES):
                with c_dfb.reserve() as c_blk:

                    # Print state of c_dfb dataflow buffer after reserve

                    print("c_dfb after reserve: ", c_dfb)

                    # Print iteration state and the content of c_blk block

                    print(
                        "i_tile=",
                        i_tile,
                        " m_tile=",
                        m_tile,
                        "n_tile=",
                        n_tile,
                        " c_blk: ",
                        c_blk,
                    )

                    c_xf = ttl.copy(c[m_tile, n_tile], c_blk)
                    c_xf.wait()

                # Print state of c_dfb dataflow buffer after push

                print("c_dfb after push: ", c_dfb)

                for k_tile in range(K_TILES):
                    with (
                        a_dfb.reserve() as a_blk,
                        b_dfb.reserve() as b_blk,
                    ):
                        # Print iteration state

                        print("k_tile=", k_tile)

                        # Print the content of a_blk block

                        print("a_blk:")
                        print(a_blk)

                        # Print the content of b_blk block

                        print("b_blk:")
                        print(b_blk)

                        a_xf = ttl.copy(a[i_tile, m_tile, k_tile], a_blk)
                        b_xf = ttl.copy(b[k_tile, n_tile], b_blk)

                        a_xf.wait()
                        b_xf.wait()


# spec:end
