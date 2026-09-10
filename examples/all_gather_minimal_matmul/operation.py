# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""All-gather + matmul + row bias.

Interface: BF16/FP32, TILE layout, DRAM tensors.
Sharding: activation along K; weight, bias, output either N-sharded or replicated.
Result: output_shard = all_gather(activation_shard) @ weight_shard + bias_shard

Per device, three kernels execute concurrently::

    receive_activations_and_write_output:
        for each M block, N block:
            if first N block or activation reuse is disabled:
                receive remote K blocks into L1
                broadcast gathered K blocks across N workers
            else:
                republish cached full-K activation blocks
            write completed output block to DRAM

    send_activations_and_broadcast_weights:
        for each M block, N block:
            if first N block or activation reuse is disabled:
                send this device's activation K blocks to peer devices
                with ring selection: forward received blocks to the next device
            broadcast weight K blocks across M workers
            read row bias

    compute_matmul_and_bias:
        for each M block, N block:
            accumulator = 0
            for each K block from every device:
                wait for activation and weight blocks
                accumulator += activation_block @ weight_block
            output_block = cast(accumulator + broadcast(row_bias))

Implementations: ``per_row_all_gather/operation.py``,
``two_worker_ring/operation.py`` and ``dedicated_communication/operation.py``.
Each defines ``all_gather_minimal_matmul`` below its network configuration.

Run from the repository root with the selected devices idle::

    # One device: identity all-gather; fabric disabled.
    python -m examples.all_gather_minimal_matmul --mesh-shape 1x1

    # Two devices: use 2x1 instead when that is the connected orientation.
    python -m examples.all_gather_minimal_matmul --mesh-shape 1x2

    # Four devices.
    python -m examples.all_gather_minimal_matmul --mesh-shape 2x2

    # Four devices, 120 compute and four communication workers/device, N-sharded output.
    python -m examples.all_gather_minimal_matmul.n_sharded --mesh-shape 2x2 \
        --worker-grid 12 10 --transpose --dedicated-communication-workers 4 \
        --m-tiles 24 --k-tiles-per-device 4 --n-tiles-per-device 20 \
        --m-block-tiles 2 --k-block-tiles 2 --n-block-tiles 2 \
        --no-reuse-activation --activation-all-gather ring

Dimensions and mesh selection: ``README.md`` beside this file.
"""

from collections.abc import Callable

from .config import AllGatherMinimalMatmulConfig
from .dedicated_communication.operation import (
    make_all_gather_minimal_matmul_operation as make_dedicated_operation,
)
from .per_row_all_gather.operation import (
    make_all_gather_minimal_matmul_operation as make_per_row_operation,
)
from .two_worker_ring.operation import (
    make_all_gather_minimal_matmul_operation as make_two_worker_ring_operation,
)


def make_all_gather_minimal_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    all_gather_algorithm: str = "all_to_all",
    dedicated_communication_workers: int | None = None,
) -> Callable[..., None]:
    """Select co-located or dedicated activation communication workers."""
    if dedicated_communication_workers is not None:
        return make_dedicated_operation(
            config,
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            all_gather_algorithm=all_gather_algorithm,
            communication_worker_count=dedicated_communication_workers,
        )
    factory = make_per_row_operation
    if (
        all_gather_algorithm == "ring"
        and config.device_count > 1
        and config.m_workers > 2
    ):
        factory = make_two_worker_ring_operation
    return factory(
        config,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        all_gather_algorithm=all_gather_algorithm,
    )
