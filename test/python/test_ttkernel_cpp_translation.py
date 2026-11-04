# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for translating multiple TTKernel functions to C++."""

import pytest

import ttl.dialects.ttl as ttl_dialect
from ttl.ir import Context, MLIRError, Module
from ttl.passes import ttkernels_to_cpp


def test_ttkernels_to_cpp_preserves_requested_order():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context:
        module = Module.parse(
            """
            module {
              func.func @reader() attributes {
                ttkernel.thread = #ttkernel.thread<noc>
              } {
                %value = arith.constant 1 : i32
                return
              }
              func.func @writer() attributes {
                ttkernel.thread = #ttkernel.thread<noc>
              } {
                %value = arith.constant 2 : i32
                return
              }
            }
            """
        )

        writer_source, reader_source = ttkernels_to_cpp(module, ["writer", "reader"])

    assert " = 2;" in writer_source
    assert " = 1;" not in writer_source
    assert " = 1;" in reader_source
    assert " = 2;" not in reader_source


def test_ttkernels_to_cpp_reports_conversion_diagnostic():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context:
        module = Module.parse(
            """
            module {
              func.func @unsupported_float() attributes {
                ttkernel.thread = #ttkernel.thread<compute>
              } {
                %value = arith.constant 0.0 : f16
                return
              }
            }
            """
        )

        with pytest.raises(MLIRError) as error:
            ttkernels_to_cpp(module, ["unsupported_float"])

    message = str(error.value)
    assert "TTKernel-to-EmitC conversion failed" in message
    assert "'emitc.constant' op requires attribute" in message


def test_ttkernels_to_cpp_emits_topk_api():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context:
        module = Module.parse(
            """
            module {
              func.func @topk() attributes {
                ttkernel.thread = #ttkernel.thread<compute>
              } {
                %dst = arith.constant 0 : index
                %direction = arith.constant 0 : i32
                %end_phase = arith.constant 4 : i32
                %start_phase = arith.constant 0 : i32
                %iteration = arith.constant 0 : i32
                %k = arith.constant 32 : i32
                %logk = arith.constant 5 : i32
                %skip_second = arith.constant 1 : i32
                ttkernel.topk_tile_init() : () -> ()
                ttkernel.topk_local_sort(
                  %dst, %direction, %end_phase, %start_phase
                ) : (index, i32, i32, i32) -> ()
                ttkernel.topk_merge(
                  %dst, %iteration, %k
                ) : (index, i32, i32) -> ()
                ttkernel.topk_rebuild(
                  %dst, %direction, %iteration, %k, %logk, %skip_second
                ) : (index, i32, i32, i32, i32, i32) -> ()
                return
              }
            }
            """
        )

        (source,) = ttkernels_to_cpp(module, ["topk"])

    assert '#include "api/compute/topk.h"' in source
    assert "topk_tile_init();" in source
    assert "topk_local_sort(" in source
    assert "topk_merge(" in source
    assert "topk_rebuild(" in source


def test_ttkernels_to_cpp_emits_topk_template_arguments():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context:
        module = Module.parse(
            """
            module {
              func.func @topk_args() attributes {
                ttkernel.thread = #ttkernel.thread<compute>
              } {
                %dst = arith.constant 0 : index
                %direction = arith.constant 1 : i32
                %end_phase = arith.constant 4 : i32
                %start_phase = arith.constant 4 : i32
                %end_step = arith.constant 6 : i32
                %start_step = arith.constant 4 : i32
                %iteration = arith.constant 1 : i32
                %k = arith.constant 32 : i32
                %logk = arith.constant 5 : i32
                %skip_second = arith.constant 0 : i32
                ttkernel.topk_tile_init() {rank_stamped = true, tag_bits = 8 : i32}
                    : () -> ()
                ttkernel.topk_local_sort(
                  %dst, %direction, %end_phase, %start_phase, %end_step, %start_step
                ) {fp32_dest_acc_en = true, stable_sort = true,
                   tie_order = #ttkernel.topk_tie_order<descending>}
                    : (index, i32, i32, i32, i32, i32) -> ()
                ttkernel.topk_merge(%dst, %iteration, %k)
                    {direction = true}
                    : (index, i32, i32) -> ()
                ttkernel.topk_rebuild(
                  %dst, %direction, %iteration, %k, %logk, %skip_second
                ) {fused = true}
                    : (index, i32, i32, i32, i32, i32) -> ()
                return
              }
            }
            """
        )

        (source,) = ttkernels_to_cpp(module, ["topk_args"])

    assert "topk_tile_init<false, true, 8>();" in source
    assert (
        "topk_local_sort<true, true, false, false, TopkTieOrder::Descending>(" in source
    )
    assert "topk_merge<true>(" in source
    assert "topk_rebuild<false, DST_ACCUM_MODE, true>(" in source
