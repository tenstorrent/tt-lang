import pytest
import torch
import ttnn

from examples.flash_chain_8node import (
    HEAD_DIM,
    HEAD_DIM_V,
    KERNEL_CONFIG_BUFFER_SIZE,
    PCC_THRESHOLD,
    Q_ROWS,
    SCALE,
    SEQ,
    flash_chain_8node,
)
from ttlang_test_utils import is_hardware_available, to_dram
from utils.correctness import assert_pcc


@pytest.fixture(scope="module")
def flash_device():
    """Provide enough kernel-config L1 for the fused compute program."""
    if not is_hardware_available():
        pytest.skip("No Tenstorrent device available")

    max_worker_l1_size = ttnn.device.get_max_worker_l1_unreserved_size()
    worker_l1_size = max_worker_l1_size - KERNEL_CONFIG_BUFFER_SIZE
    device = ttnn.open_device(device_id=0, worker_l1_size=worker_l1_size)
    yield device
    ttnn.close_device(device)


def test_flash_chain_8node(flash_device):
    torch.manual_seed(2026)
    query_host = torch.randn(Q_ROWS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    key_host = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    value_host = torch.randn(SEQ, HEAD_DIM_V, dtype=torch.bfloat16) * 0.1
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query_host.float().unsqueeze(0).unsqueeze(0),
            key_host.float().unsqueeze(0).unsqueeze(0),
            value_host.float().unsqueeze(0).unsqueeze(0),
            scale=SCALE,
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bfloat16)
    )

    query = to_dram(query_host, flash_device)
    key = to_dram(key_host, flash_device)
    value = to_dram(value_host, flash_device)
    output = to_dram(
        torch.zeros(Q_ROWS, HEAD_DIM_V, dtype=torch.bfloat16),
        flash_device,
    )

    flash_chain_8node(query, key, value, output)

    actual = ttnn.to_torch(output).reshape(Q_ROWS, HEAD_DIM_V).float()
    assert_pcc(expected.float(), actual, threshold=PCC_THRESHOLD)
