import ttnn
import ttl


header_path = "miscHeader.hpp"
call_operator = "dummy_print()"
args = []


@ttl.operation(grid = (1,1))
def dummy_print(out):
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def fill_compute():
        with out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(1.0, shape=out_blk.shape))
        #call_extern_func()

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

device = ttnn.open_device(device_id=0)
out = ttnn.empty(shape=[32, 32], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
dummy_print(out)
ttnn.close_device(device)
