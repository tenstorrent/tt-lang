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
    for i_tile in range(I_TILES):
        for m_tile in range(M_TILES):
            for n_tile in range(N_TILES):

                # Measure the entire iteration

                with ttl.signpost("i_m_n iteration"):

                    # Measure from reserve to push

                    with ttl.signpost("push c"):
                        with c_dfb.reserve() as c_blk:

                            # Measure only copy

                            with ttl.signpost("read c"):
                                c_xf = ttl.copy(c[m_tile, n_tile], c_blk)
                                c_xf.wait()

                    for k_tile in range(K_TILES):
                        with ttl.signpost("push a and b"):

                            # Measure from reserve to push

                            with (
                                a_dfb.reserve() as a_blk,
                                b_dfb.reserve() as b_blk,
                            ):

                                # Measure only copy

                                with ttl.signpost("read a and b"):
                                    a_xf = ttl.copy(a[i_tile, m_tile, k_tile], a_blk)
                                    b_xf = ttl.copy(b[k_tile, n_tile], b_blk)

                                    a_xf.wait()
                                    b_xf.wait()


# spec:end
