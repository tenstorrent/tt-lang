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
