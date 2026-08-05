# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reuse DFB storage around a direct external-function multiply.

This operation has ten logical DFBs. The allocator sees the DFB call arguments
but cannot inspect the C++ reserve, wait, push, and pop operations, so the six
DFBs passed to the two external calls retain separate physical indices. The
three explicit copy intermediates and copied-product DFB form a linear copy
chain. Their proven
non-overlapping lifetimes use two physical DFB indices alternately, reducing
the complete operation to eight physical DFB indices.

This is a direct external-function example, not a tt-blaze `Op` header. Blaze
operations use typed compile-time argument structures and per-thread methods;
the tt-lang external-function interface accepts physical DFB indices instead.

Run on hardware with:

    source build/env/activate
    python examples/external_dfb_reuse.py

Set `TTLANG_FINAL_MLIR` to inspect the eight `ttl.dfb_allocations` descriptors.
"""

import os
import re
import tempfile
from pathlib import Path

import torch
import ttl
import ttnn

TILE = 32
LOGICAL_DFB_COUNT = 10
PHYSICAL_DFB_COUNT = 8
EXTERNAL_MULTIPLY_HEADER = str(
    Path(__file__).parent / "include" / "external_eltwise_mul.hpp"
)


@ttl.operation()
def external_multiply(lhs: ttl.DFB, rhs: ttl.DFB, result: ttl.DFB) -> None:
    call_extern_func(
        EXTERNAL_MULTIPLY_HEADER,
        "ttl_external_eltwise_mul",
        func_args=[lhs, rhs, result],
    )


@ttl.operation()
def copy_stage(source: ttl.DFB, destination: ttl.DFB) -> None:
    destination_block = destination.reserve()
    destination_block.store(source.wait())


@ttl.operation(grid=(1, 1))
def external_dfb_reuse(lhs, rhs, result) -> None:
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    first_rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    second_rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    product_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    copy_one_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    copy_two_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    copy_three_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    copied_product_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(1, 1), block_count=2
    )
    computed_product_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(1, 1), block_count=2
    )
    result_dfb = ttl.make_dataflow_buffer_like(result, shape=(1, 1), block_count=2)

    lhs_destination = lhs_dfb.reserve()
    ttl.copy(lhs[0, 0], lhs_destination).wait()
    lhs_destination.push()
    first_rhs_destination = first_rhs_dfb.reserve()
    ttl.copy(rhs[0, 0], first_rhs_destination).wait()
    first_rhs_destination.push()
    second_rhs_destination = second_rhs_dfb.reserve()
    ttl.copy(rhs[0, 0], second_rhs_destination).wait()
    second_rhs_destination.push()

    external_multiply(lhs_dfb, first_rhs_dfb, product_dfb)

    # Each copy releases its source before the next copy reserves its
    # destination. The four logical destinations therefore use two physical
    # DFB indices rather than requiring four separate indices.
    copy_stage(product_dfb, copy_one_dfb)
    copy_stage(copy_one_dfb, copy_two_dfb)
    copy_stage(copy_two_dfb, copy_three_dfb)
    copy_stage(copy_three_dfb, copied_product_dfb)

    copied_product = copied_product_dfb.wait()
    computed_product = computed_product_dfb.reserve()
    computed_product.store(ttl.exp(copied_product))
    computed_product.push()

    external_multiply(computed_product_dfb, second_rhs_dfb, result_dfb)

    result_source = result_dfb.wait()
    ttl.copy(result_source, result[0, 0]).wait()
    result_source.pop()


def to_device(tensor: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def count_final_dfb_allocations(final_mlir_path: Path) -> int:
    final_mlir = final_mlir_path.read_text()
    allocations = re.search(
        r"ttl\.dfb_allocations = \[(.*?)\](?:,|})", final_mlir, re.DOTALL
    )
    if allocations is None:
        raise RuntimeError("final MLIR does not contain ttl.dfb_allocations")
    return allocations.group(1).count("dfb_index =")


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(
            TILE, TILE
        )
        lhs_host = ((element_indices.remainder(41) - 20) / 16).to(torch.bfloat16)
        rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(
            torch.bfloat16
        )

        lhs = to_device(lhs_host, device)
        rhs = to_device(rhs_host, device)
        result = to_device(torch.zeros_like(lhs_host), device)

        with tempfile.TemporaryDirectory() as temporary_directory:
            final_mlir_path = Path(
                os.environ.get(
                    "TTLANG_FINAL_MLIR",
                    str(Path(temporary_directory) / "external_dfb_reuse.mlir"),
                )
            )
            previous_final_mlir = os.environ.get("TTLANG_FINAL_MLIR")
            os.environ["TTLANG_FINAL_MLIR"] = str(final_mlir_path)
            try:
                external_dfb_reuse(lhs, rhs, result, options="--ttl-reuse-user-dfbs")
            finally:
                if previous_final_mlir is None:
                    os.environ.pop("TTLANG_FINAL_MLIR", None)
                else:
                    os.environ["TTLANG_FINAL_MLIR"] = previous_final_mlir

            physical_dfb_count = count_final_dfb_allocations(final_mlir_path)
            if physical_dfb_count != PHYSICAL_DFB_COUNT:
                raise RuntimeError(
                    f"expected {PHYSICAL_DFB_COUNT} physical DFBs, got "
                    f"{physical_dfb_count}"
                )

        actual = ttnn.to_torch(result).float()
        expected = torch.exp(lhs_host.float() * rhs_host.float()) * rhs_host.float()
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=1.0)
        print(
            f"PASSED: {LOGICAL_DFB_COUNT} logical DFBs use "
            f"{physical_dfb_count} physical DFB indices."
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
