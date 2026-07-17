import ttnn
import ttl


@ttl.operation(grid = (1,1))
def dummy_print(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))
        call_extern_func("/workspace/tt-lang/miscHeader.hpp", "dummy_print")

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

@ttl.operation(grid = (1,1))
def dummy_int_add(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))
        a = 1
        b = 2
        call_extern_func("/workspace/tt-lang/miscHeader.hpp", "dummy_int_add", a, b)

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid = (1,1))
def dummy_fp_neq(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))
        a = 1.0
        b = 2.0
        call_extern_func("/workspace/tt-lang/miscHeader.hpp", "dummy_fp_neq", a, b)

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid = (1,1))
def dummy_cb_index(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))
        # Pass the dataflow buffer itself: lowering resolves it to its CB
        # index (i32) and the header receives it as `int cb`.
        call_extern_func("/workspace/tt-lang/miscHeader.hpp", "dummy_cb_index", out_dfb)

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid = (1,1))
def dump_front(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        # Inside wait(): cb_wait_front has run, so the front tile holds the
        # compute thread's data. Header reads it via get_read_ptr (dataflow
        # API, available on this Noc thread) and prints the first word.
        with out_dfb.wait() as blk:
            call_extern_func("/workspace/tt-lang/miscHeader.hpp", "dump_cb_front", out_dfb)
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


device = ttnn.open_device(device_id=0)
out = ttnn.empty(shape=[32, 32], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
dummy_print(out)
dummy_int_add(out)
dummy_fp_neq(out)
dummy_cb_index(out)
dump_front(out)
ttnn.close_device(device)
