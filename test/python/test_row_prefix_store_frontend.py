# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Frontend contract coverage for compact row-prefix stores."""

import os

import pytest

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl  # noqa: E402
from ttl.diagnostics import TTLangCompileError  # noqa: E402


def _make_valid_row_prefix_operation(data_format):
    @ttl.operation(grid=(1, 1))
    def row_prefix_operation():
        source_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=2, tile=(32, 32)
        )
        output_dfb = ttl.make_dfb(
            data_format, shape=(1, 14), block_count=1, tile=(1, 32)
        )

        @ttl.datamovement()
        def produce():
            for _produce_index in range(2):
                with source_dfb.reserve() as _source_block:
                    pass

        @ttl.compute()
        def compute():
            with output_dfb.reserve() as output_block:
                with source_dfb.wait() as source_block:
                    output_block.store_rows(source_block)
                for _accumulate_index in range(1):
                    with source_dfb.wait() as source_block:
                        output_block.accumulate_rows(source_block)

        @ttl.datamovement()
        def consume():
            with output_dfb.wait() as _output_block:
                pass

    return row_prefix_operation


def _make_invalid_store_rows_operation(
    *,
    source_data_format="bf16",
    source_shape=(1, 1),
    source_tile=(32, 32),
    destination_data_format="bf16",
    destination_shape=(1, 14),
    destination_tile=(1, 32),
):
    @ttl.operation(grid=(1, 1))
    def invalid_store_rows_operation():
        source_dfb = ttl.make_dfb(
            source_data_format,
            shape=source_shape,
            block_count=1,
            tile=source_tile,
        )
        destination_dfb = ttl.make_dfb(
            destination_data_format,
            shape=destination_shape,
            block_count=1,
            tile=destination_tile,
        )

        @ttl.compute()
        def compute():
            with source_dfb.wait() as source_block:
                with destination_dfb.reserve() as destination_block:
                    destination_block.store_rows(source_block)

    return invalid_store_rows_operation


@ttl.operation(grid=(1, 1))
def _store_rows_from_waited_destination():
    source_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=1, tile=(32, 32))
    destination_dfb = ttl.make_dfb("bf16", shape=(1, 14), block_count=1, tile=(1, 32))

    @ttl.compute()
    def compute():
        with source_dfb.wait() as source_block:
            with destination_dfb.wait() as destination_block:
                destination_block.store_rows(source_block)


@ttl.operation(grid=(1, 1))
def _accumulate_rows_into_waited_destination():
    source_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=1, tile=(32, 32))
    destination_dfb = ttl.make_dfb("bf16", shape=(1, 14), block_count=1, tile=(1, 32))

    @ttl.compute()
    def compute():
        with source_dfb.wait() as source_block:
            with destination_dfb.wait() as destination_block:
                destination_block.accumulate_rows(source_block)


@ttl.operation(grid=(1, 1))
def _store_rows_from_scalar():
    destination_dfb = ttl.make_dfb("bf16", shape=(1, 14), block_count=1, tile=(1, 32))

    @ttl.compute()
    def compute():
        node_x, _node_y = ttl.node(dims=2)
        with destination_dfb.reserve() as destination_block:
            destination_block.store_rows(node_x)


INVALID_ROW_PREFIX_CASES = [
    pytest.param(
        _store_rows_from_waited_destination,
        r"store_rows\(\) requires a reserve-backed block",
        id="store-wait-backed",
    ),
    pytest.param(
        _accumulate_rows_into_waited_destination,
        r"accumulate_rows\(\) requires a reserve-backed block",
        id="accumulate-wait-backed",
    ),
    pytest.param(
        _store_rows_from_scalar,
        "row-prefix store requires ranked tensor operands",
        id="non-tensor-source",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(source_tile=(16, 32)),
        "row-prefix store source must use 32x32 tiles, got 16x32",
        id="source-tile-shape",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(source_shape=(1, 2)),
        r"row-prefix store source must contain exactly one tile, got shape \(1, 2\)",
        id="source-tile-count",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(destination_data_format="f32"),
        "row-prefix store source and destination dtypes must match",
        id="dtype-mismatch",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(
            source_data_format="i32", destination_data_format="i32"
        ),
        "row-prefix store supports only bf16 and f32 tile data types",
        id="unsupported-dtype",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(
            destination_shape=(1, 28), destination_tile=(1, 16)
        ),
        "row-prefix store destination tile width must equal source width 32, got 16",
        id="destination-width",
    ),
    pytest.param(
        _make_invalid_store_rows_operation(destination_shape=(1, 33)),
        "row-prefix store destination must contain between 1 and 1024 scalar "
        "elements, got 1056",
        id="destination-capacity",
    ),
]


@pytest.mark.parametrize("data_format", ["bf16", "f32"])
def test_row_prefix_methods_emit_expected_store_attributes(
    data_format, tmp_path, monkeypatch
):
    initial_mlir = tmp_path / f"row_prefix_{data_format}.mlir"
    monkeypatch.setenv("TTLANG_INITIAL_MLIR", str(initial_mlir))

    _make_valid_row_prefix_operation(data_format)()

    initial_text = initial_mlir.read_text()
    assert f"!ttcore.tile<32x32, {data_format}>" in initial_text
    assert f"!ttcore.tile<1x32, {data_format}>" in initial_text
    assert "ttl.store" in initial_text
    assert "{row_prefix}" in initial_text
    assert "{accumulate, row_prefix}" in initial_text


@pytest.mark.parametrize(("operation", "diagnostic"), INVALID_ROW_PREFIX_CASES)
def test_row_prefix_methods_reject_invalid_frontend_calls(operation, diagnostic):
    with pytest.raises(TTLangCompileError, match=diagnostic):
        operation()
