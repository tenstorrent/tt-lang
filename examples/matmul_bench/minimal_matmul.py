# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Multicore blocked matmul structured to mirror the tt-metal minimal_matmul
# compute kernel:
#   tt-metal: ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp
#   dataflow: ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp
#             ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp
#   test:     tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py
#
# tt-metal minimal_matmul loop structure:
#
#   for m_block in M_blocks_per_core:
#     for n_block in N_blocks_per_core:
#       cb_reserve_back(intermediate_cb)
#       for k_block in K_num_blocks:
#         matmul_blocks(in0, in1, intermediate, M_block, N_block, K_block, ...)
#         if k_block == 0: pack_reconfig_l1_acc(1)
#       cb_push_back(intermediate_cb)
#       pack_reconfig_l1_acc(0)
#       // post-ops: copy_block or add_bias_block
#
# tt-lang maps this as:
#   - Each compute invocation handles one (M_block, N_block) output block
#     with K_block inner K tiles. The compiler generates the K loop and
#     subblocking from the 3D [M, N, K] iteration space.
#   - Subblocking within matmul_blocks: auto-managed by TTLSubblockComputeForDST.
#   - L1 accumulation across K_num_blocks: compiler-inserted via
#     TTKernelInsertL1Accumulation when the K reduction loop is present.
#   - The user-written outer K loop streams K blocks from DRAM and passes
#     each K_block-sized chunk to a single matmul expression.
#   - Post-ops: explicit elementwise add for bias, applied after K reduction.
#   - Work distribution: 2D grid with M on grid_y, N on grid_x.
#
# Features NOT yet expressible in tt-lang (compiler/runtime limitations):
#   - K-direction alternation (k_forward flag) for in0 reuse across N blocks.
#     Requires runtime ternary expressions and min(); see runtime-expressions plan.
#   - Deferred output writes overlapped with next block's DMA.
#
# tt-metal benchmark reference:
#   4096x4096x4096 with M_block=8, K_block=8, N_block=8, subblock_h=2, subblock_w=2
#   = 128x128x128 tiles, K_num_blocks=16

import ttl
import ttnn
from utils.correctness import assert_pcc


TILE = 32


def make_minimal_matmul(M_block_tiles, K_block_tiles, N_block_tiles, fp32_acc=None):
    """Matmul without bias: out = a @ b.

    Matches tt-metal minimal_matmul with FUSE_BIAS=0.

    Each compute invocation receives one K_block-sized chunk per input.
    The compiler generates the inner K reduction loop and subblocking from
    the 3D [M_block, N_block, K_block] iteration space. The user manages the
    outer K_num_blocks loop for streaming K blocks from DRAM.

    When M_block * N_block > DST capacity, TTLSubblockComputeForDST tiles
    the parallel dims. TTKernelInsertL1Accumulation inserts pack_reconfig_l1_acc
    guards around the K reduction loop.

    Args:
        fp32_acc: If True, use f32 DST accumulation (DST capacity halved).
                  If False, use bf16 accumulation (DST capacity doubled).
                  If None, auto-detect from input dtype.
    """

    @ttl.operation(grid="auto", fp32_dest_acc_en=fp32_acc)
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        K_num_blocks = Kt // K_block_tiles
        M_num_blocks = Mt // M_block_tiles
        N_num_blocks = Nt // N_block_tiles

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_blocks_per_node = -(-M_num_blocks // grid_m)
        n_blocks_per_node = -(-N_num_blocks // grid_n)

        # in0: A block [M_block, K_block]. Double-buffered for DMA/compute overlap.
        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(M_block_tiles, K_block_tiles), block_count=2
        )
        # in1: B block [K_block, N_block].
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(K_block_tiles, N_block_tiles), block_count=2
        )
        # intermediate: accumulator for K reduction. Compute-local (DM does not touch).
        acc_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )
        # out: final output, written by DM writer.
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            # First K block: standalone matmul.
                            # The compiler generates the inner K_block reduction
                            # from the 3D [M_block, N_block, K_block] iteration
                            # space of a @ b.
                            a_blk = a_dfb.wait()
                            b_blk = b_dfb.wait()
                            with acc_dfb.reserve() as acc:
                                acc.store(a_blk @ b_blk)
                            a_blk.pop()
                            b_blk.pop()

                            # Remaining K blocks: accumulate via prev + a @ b.
                            # Lowers to copy_tile(prev) + matmul_block(DST += A*B).
                            for _ in range(K_num_blocks - 1):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as prev,
                                ):
                                    with acc_dfb.reserve() as acc:
                                        acc.store(prev + a_blk @ b_blk)

                            # Copy to output (matches copy_block in tt-metal).
                            with acc_dfb.wait() as acc_blk:
                                with out_dfb.reserve() as out_blk:
                                    out_blk.store(acc_blk)

        # Split DMA across both RISC cores (matches tt-metal):
        #   reader (NCRISC): reads A blocks only
        #   writer (BRISC):  reads B blocks + writes output
        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with a_dfb.reserve() as a_blk:
                                    ttl.copy(
                                        a[
                                            m_off : m_off + M_block_tiles,
                                            k_off : k_off + K_block_tiles,
                                        ],
                                        a_blk,
                                    ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            # Read B blocks for this output block.
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with b_dfb.reserve() as b_blk:
                                    ttl.copy(
                                        b[
                                            k_off : k_off + K_block_tiles,
                                            n_off : n_off + N_block_tiles,
                                        ],
                                        b_blk,
                                    ).wait()
                            # Write output after K reduction completes.
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                ).wait()

    return kernel


def make_minimal_matmul_single_reader(
    M_block_tiles, K_block_tiles, N_block_tiles, fp32_acc=None
):
    """Matmul without bias, single-reader DMA: out = a @ b.

    Same compute kernel as make_minimal_matmul, but all DMA reads (A and B)
    are on one RISC core (reader). The writer thread only writes output.
    This is the v1-single-reader variant used to measure the effect of
    splitting DMA across both RISC cores.
    """

    @ttl.operation(grid="auto", fp32_dest_acc_en=fp32_acc)
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        K_num_blocks = Kt // K_block_tiles
        M_num_blocks = Mt // M_block_tiles
        N_num_blocks = Nt // N_block_tiles

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_blocks_per_node = -(-M_num_blocks // grid_m)
        n_blocks_per_node = -(-N_num_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(M_block_tiles, K_block_tiles), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(K_block_tiles, N_block_tiles), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            a_blk = a_dfb.wait()
                            b_blk = b_dfb.wait()
                            with acc_dfb.reserve() as acc:
                                acc.store(a_blk @ b_blk)
                            a_blk.pop()
                            b_blk.pop()

                            for _ in range(K_num_blocks - 1):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as prev,
                                ):
                                    with acc_dfb.reserve() as acc:
                                        acc.store(prev + a_blk @ b_blk)

                            with acc_dfb.wait() as acc_blk:
                                with out_dfb.reserve() as out_blk:
                                    out_blk.store(acc_blk)

        # Single-reader DMA: one thread reads both A and B.
        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with a_dfb.reserve() as a_blk:
                                    ttl.copy(
                                        a[
                                            m_off : m_off + M_block_tiles,
                                            k_off : k_off + K_block_tiles,
                                        ],
                                        a_blk,
                                    ).wait()
                                with b_dfb.reserve() as b_blk:
                                    ttl.copy(
                                        b[
                                            k_off : k_off + K_block_tiles,
                                            n_off : n_off + N_block_tiles,
                                        ],
                                        b_blk,
                                    ).wait()

        # Writer only writes output.
        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                ).wait()

    return kernel


def make_minimal_matmul_with_bias(M_block_tiles, K_block_tiles, N_block_tiles):
    """Matmul with bias: out = a @ b + c.

    Matches tt-metal minimal_matmul with FUSE_BIAS=1.
    Bias is added after K reduction (same ordering as tt-metal's add_bias_block).
    """

    @ttl.operation(grid="auto")
    def kernel(a, b, bias, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        K_num_blocks = Kt // K_block_tiles
        M_num_blocks = Mt // M_block_tiles
        N_num_blocks = Nt // N_block_tiles

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_blocks_per_node = -(-M_num_blocks // grid_m)
        n_blocks_per_node = -(-N_num_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(M_block_tiles, K_block_tiles), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(K_block_tiles, N_block_tiles), block_count=2
        )
        # in2: bias block [M_block, N_block]. Matches tt-metal's in2_cb / cb_id_in2.
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias, shape=(M_block_tiles, N_block_tiles), block_count=2
        )
        acc_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            # K reduction loop.
                            a_blk = a_dfb.wait()
                            b_blk = b_dfb.wait()
                            with acc_dfb.reserve() as acc:
                                acc.store(a_blk @ b_blk)
                            a_blk.pop()
                            b_blk.pop()

                            for _ in range(K_num_blocks - 1):
                                with (
                                    a_dfb.wait() as a_blk,
                                    b_dfb.wait() as b_blk,
                                    acc_dfb.wait() as prev,
                                ):
                                    with acc_dfb.reserve() as acc:
                                        acc.store(prev + a_blk @ b_blk)

                            # Bias add (matches add_bias_block in tt-metal).
                            with (
                                acc_dfb.wait() as acc_blk,
                                bias_dfb.wait() as bias_blk,
                            ):
                                with out_dfb.reserve() as out_blk:
                                    out_blk.store(bias_blk + acc_blk)

        # Split DMA: reader handles A, writer handles B + bias + output.
        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with a_dfb.reserve() as a_blk:
                                    ttl.copy(
                                        a[
                                            m_off : m_off + M_block_tiles,
                                            k_off : k_off + K_block_tiles,
                                        ],
                                        a_blk,
                                    ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with b_dfb.reserve() as b_blk:
                                    ttl.copy(
                                        b[
                                            k_off : k_off + K_block_tiles,
                                            n_off : n_off + N_block_tiles,
                                        ],
                                        b_blk,
                                    ).wait()
                            # Bias after K blocks.
                            with bias_dfb.reserve() as bias_blk:
                                ttl.copy(
                                    bias[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                    bias_blk,
                                ).wait()
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                ).wait()

    return kernel


def make_matmul_compiler_k_loop(M_block_tiles, N_block_tiles, fp32_acc=None):
    """Matmul with compiler-generated K loop: out = a @ b.

    Full K dimension is in the DFB. The compiler generates the K reduction
    loop from the 3D [M_block, N_block, K] iteration space. When
    M_block * N_block > DST capacity, this requires the hybrid pattern:
    subblocked M/N with L1 accumulation across K.

    This is the target codegen pattern for the L1 accumulation plan.
    """

    @ttl.operation(grid="auto", fp32_dest_acc_en=fp32_acc)
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        M_num_blocks = Mt // M_block_tiles
        N_num_blocks = Nt // N_block_tiles

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_blocks_per_node = -(-M_num_blocks // grid_m)
        n_blocks_per_node = -(-N_num_blocks // grid_n)

        # Full K in DFB: compiler handles K reduction internally.
        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(M_block_tiles, Kt), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(Kt, N_block_tiles), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            # Single matmul expression: compiler generates
                            # the K reduction loop and L1 acc guards.
                            a_blk = a_dfb.wait()
                            b_blk = b_dfb.wait()
                            with out_dfb.reserve() as out_blk:
                                out_blk.store(a_blk @ b_blk)
                            a_blk.pop()
                            b_blk.pop()

        # Split DMA: reader handles A, writer handles B + output.
        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            with a_dfb.reserve() as a_blk:
                                ttl.copy(
                                    a[m_off : m_off + M_block_tiles, 0:Kt],
                                    a_blk,
                                ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            with b_dfb.reserve() as b_blk:
                                ttl.copy(
                                    b[0:Kt, n_off : n_off + N_block_tiles],
                                    b_blk,
                                ).wait()
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                ).wait()

    return kernel


def make_matmul_l1_acc(M_block_tiles, K_block_tiles, N_block_tiles, fp32_acc=None):
    """Matmul with L1 accumulation: out = a @ b.

    Uses the "reserve once, store K times, push once" pattern. The compiler
    detects the scf.for loop storing to the same reserved CB and annotates
    it as a reduction loop. TTKernelInsertL1Accumulation inserts
    pack_reconfig_l1_acc guards. Each K iteration packs to L1 additively,
    eliminating the copy_tile + acc_dfb overhead of the prev + a @ b pattern.

    DMA is split: reader (NCRISC) handles A, writer (BRISC) handles B + output.
    """

    @ttl.operation(grid="auto", fp32_dest_acc_en=fp32_acc)
    def kernel(a, b, out):
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = b.shape[1] // TILE

        K_num_blocks = Kt // K_block_tiles
        M_num_blocks = Mt // M_block_tiles
        N_num_blocks = Nt // N_block_tiles

        grid_n, grid_m = ttl.grid_size(dims=2)
        m_blocks_per_node = -(-M_num_blocks // grid_m)
        n_blocks_per_node = -(-N_num_blocks // grid_n)

        a_dfb = ttl.make_dataflow_buffer_like(
            a, shape=(M_block_tiles, K_block_tiles), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b, shape=(K_block_tiles, N_block_tiles), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(M_block_tiles, N_block_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            # Reserve output once before K loop.
                            out_blk = out_dfb.reserve()
                            # K loop: each store packs to same CB slot.
                            # L1 acc makes subsequent packs additive.
                            for _ in range(K_num_blocks):
                                a_blk = a_dfb.wait()
                                b_blk = b_dfb.wait()
                                out_blk.store(a_blk @ b_blk)
                                a_blk.pop()
                                b_blk.pop()
                            # Push once after all K iterations.
                            out_blk.push()

        @ttl.datamovement()
        def reader():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with a_dfb.reserve() as a_blk:
                                    ttl.copy(
                                        a[
                                            m_off : m_off + M_block_tiles,
                                            k_off : k_off + K_block_tiles,
                                        ],
                                        a_blk,
                                    ).wait()

        @ttl.datamovement()
        def writer():
            node_n, node_m = ttl.node(dims=2)
            for local_m in range(m_blocks_per_node):
                m_block = node_m * m_blocks_per_node + local_m
                if m_block < M_num_blocks:
                    m_off = m_block * M_block_tiles
                    for local_n in range(n_blocks_per_node):
                        n_block = node_n * n_blocks_per_node + local_n
                        if n_block < N_num_blocks:
                            n_off = n_block * N_block_tiles
                            for kb in range(K_num_blocks):
                                k_off = kb * K_block_tiles
                                with b_dfb.reserve() as b_blk:
                                    ttl.copy(
                                        b[
                                            k_off : k_off + K_block_tiles,
                                            n_off : n_off + N_block_tiles,
                                        ],
                                        b_blk,
                                    ).wait()
                            with out_dfb.wait() as out_blk:
                                ttl.copy(
                                    out_blk,
                                    out[
                                        m_off : m_off + M_block_tiles,
                                        n_off : n_off + N_block_tiles,
                                    ],
                                ).wait()

    return kernel


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

import torch


def test_matmul(device, Mt, Kt, Nt, M_block, K_block, N_block):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    print(
        f"  matmul [{M},{K}] @ [{K},{N}] "
        f"blocks=({M_block},{K_block},{N_block}) "
        f"K_num_blocks={Kt // K_block}: ",
        end="",
        flush=True,
    )

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    kernel = make_minimal_matmul(M_block, K_block, N_block)
    kernel(a, b, out)

    golden = (a_torch.float() @ b_torch.float()).float()
    assert_pcc(golden, ttnn.to_torch(out).float(), threshold=0.99)
    print("PASSED!")


def test_matmul_compiler_k(device, Mt, Kt, Nt, M_block, N_block):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    print(
        f"  compiler-K matmul [{M},{K}] @ [{K},{N}] "
        f"blocks=({M_block},{N_block}) Kt={Kt}: ",
        end="",
        flush=True,
    )

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    kernel = make_matmul_compiler_k_loop(M_block, N_block)
    kernel(a, b, out)

    golden = (a_torch.float() @ b_torch.float()).float()
    assert_pcc(golden, ttnn.to_torch(out).float(), threshold=0.99)
    print("PASSED!")


def test_matmul_with_bias(device, Mt, Kt, Nt, M_block, K_block, N_block):
    M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
    print(
        f"  matmul+bias [{M},{K}] @ [{K},{N}] + [{M},{N}] "
        f"blocks=({M_block},{K_block},{N_block}) "
        f"K_num_blocks={Kt // K_block}: ",
        end="",
        flush=True,
    )

    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)
    c_torch = torch.randn(M, N, dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    c = ttnn.from_torch(
        c_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    kernel = make_minimal_matmul_with_bias(M_block, K_block, N_block)
    kernel(a, b, c, out)

    golden = (a_torch.float() @ b_torch.float() + c_torch.float()).float()
    assert_pcc(golden, ttnn.to_torch(out).float(), threshold=0.99)
    print("PASSED!")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        # --- Matmul without bias ---
        print("Matmul (no bias):")

        # K_block=1, single K block (K_num_blocks=1).
        test_matmul(device, Mt=2, Kt=1, Nt=2, M_block=2, K_block=1, N_block=2)

        # K_block=1, multiple K blocks (K_num_blocks=4).
        test_matmul(device, Mt=4, Kt=4, Nt=4, M_block=2, K_block=1, N_block=2)

        # K_block=2 inner, 2 outer K blocks. Matches tt-metal K_block_tiles > 1.
        test_matmul(device, Mt=4, Kt=4, Nt=4, M_block=2, K_block=2, N_block=2)

        # K_block=4 inner, 4 outer K blocks. Exercises deeper K accumulation.
        test_matmul(device, Mt=8, Kt=16, Nt=8, M_block=2, K_block=4, N_block=2)

        # Larger output blocks requiring auto-subblocking (M*N > 8 DST tiles).
        test_matmul(device, Mt=8, Kt=4, Nt=8, M_block=4, K_block=2, N_block=4)

        # --- Compiler-generated K loop (full K in DFB) ---
        # Output fits in DST: works today via DST accumulation.
        print("\nCompiler K loop (output fits in DST):")
        test_matmul_compiler_k(device, Mt=2, Kt=2, Nt=2, M_block=2, N_block=2)
        test_matmul_compiler_k(device, Mt=4, Kt=4, Nt=4, M_block=2, N_block=2)

        # --- L1 accumulation cases (output > DST capacity + K > 1) ---
        # These require the hybrid pattern: subblocked M/N with L1 acc across K.
        # Output > 8 bf16 DST tiles AND K > 1 = needs subblocking + L1 acc.
        print("\nCompiler K loop + L1 acc (output > DST + K > 1):")

        # 3x3=9 > 8 DST tiles, K=2.
        test_matmul_compiler_k(device, Mt=3, Kt=2, Nt=3, M_block=3, N_block=3)

        # 4x4=16 > 8, K=4.
        test_matmul_compiler_k(device, Mt=4, Kt=4, Nt=4, M_block=4, N_block=4)

        # 8x8=64 > 8, K=8, multiple output blocks.
        test_matmul_compiler_k(device, Mt=8, Kt=8, Nt=8, M_block=4, N_block=4)

        # --- Matmul with bias (FUSE_BIAS=1) ---
        print("\nMatmul with bias:")

        # Small case.
        test_matmul_with_bias(device, Mt=2, Kt=2, Nt=2, M_block=2, K_block=1, N_block=2)

        # Multi-tile K_block with outer K accumulation + bias.
        test_matmul_with_bias(device, Mt=4, Kt=4, Nt=4, M_block=2, K_block=2, N_block=2)

        # Larger shape comparable to minimal_matmul performance tests.
        test_matmul_with_bias(
            device, Mt=8, Kt=16, Nt=8, M_block=2, K_block=4, N_block=2
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
