import ttnn
import pytest
import torch


# need to figure out what device is
def test_singlecore_matmul(device): 
    M, K, N = 640, 640, 640

    
    num_tiles = 4
    src_bank_id = 0
    dst_bank_id = 0

    shape = [1, num_tiles, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    a_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    b_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )