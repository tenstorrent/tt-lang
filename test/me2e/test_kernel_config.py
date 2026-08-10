# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for compiler-selected compute configuration propagation."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl.dialects.ttl as ttl  # noqa: E402
from ttl.ir import Context, Module  # noqa: E402

from .builder import kernels as kernel_builder  # noqa: E402
from .builder.kernels import KernelSpec, ThreadType  # noqa: E402
from .builder.ttnn_runner import _get_compute_config  # noqa: E402


def test_compiled_kernel_configuration_is_extracted(monkeypatch):
    context = Context()
    ttl.ensure_dialects_registered(context)

    with context:
        module = Module.parse(
            """
            module {
              func.func @compute_kernel() attributes {
                dst_full_sync_en = true,
                fp32_dest_acc_en = true,
                ttl.unpack_to_dest_fp32 = array<i32: 3, 17>
              } {
                return
              }
            }
            """,
            context,
        )
        monkeypatch.setattr(
            kernel_builder,
            "get_ttkernel_names",
            lambda compiled_module: [("compute_kernel", "compute")],
        )
        monkeypatch.setattr(
            kernel_builder,
            "ttkernel_to_cpp_by_name",
            lambda compiled_module, kernel_name: "void kernel_main() {}",
        )

        noc_kernels, compute_kernel = kernel_builder.translate_module_to_kernels(module)

    assert noc_kernels == []
    assert compute_kernel.fp32_dest_acc_en
    assert compute_kernel.dst_full_sync_en
    assert compute_kernel.unpack_to_dest_fp32 == [3, 17]


def test_missing_compiler_configuration_is_rejected(monkeypatch):
    context = Context()
    ttl.ensure_dialects_registered(context)

    with context:
        module = Module.parse(
            """
            module {
              func.func @compute_kernel() attributes {
                dst_full_sync_en = false,
                fp32_dest_acc_en = false
              } {
                return
              }
            }
            """,
            context,
        )
        monkeypatch.setattr(
            kernel_builder,
            "get_ttkernel_names",
            lambda compiled_module: [("compute_kernel", "compute")],
        )
        monkeypatch.setattr(
            kernel_builder,
            "ttkernel_to_cpp_by_name",
            lambda compiled_module, kernel_name: "void kernel_main() {}",
        )

        with pytest.raises(
            ValueError,
            match="Required compiler-generated attribute "
            "'ttl.unpack_to_dest_fp32' is missing",
        ):
            kernel_builder.translate_module_to_kernels(module)


def test_noc_kernel_does_not_require_compute_configuration(monkeypatch):
    context = Context()
    ttl.ensure_dialects_registered(context)

    with context:
        module = Module.parse(
            """
            module {
              func.func @reader() {
                return
              }
              func.func @compute_kernel() attributes {
                dst_full_sync_en = false,
                fp32_dest_acc_en = false,
                ttl.unpack_to_dest_fp32 = array<i32>
              } {
                return
              }
            }
            """,
            context,
        )
        monkeypatch.setattr(
            kernel_builder,
            "get_ttkernel_names",
            lambda compiled_module: [
                ("reader", "noc"),
                ("compute_kernel", "compute"),
            ],
        )
        monkeypatch.setattr(
            kernel_builder,
            "ttkernel_to_cpp_by_name",
            lambda compiled_module, kernel_name: "void kernel_main() {}",
        )

        noc_kernels, compute_kernel = kernel_builder.translate_module_to_kernels(module)

    assert len(noc_kernels) == 1
    assert noc_kernels[0].name == "reader"
    assert compute_kernel.name == "compute_kernel"


def test_compiled_kernel_configuration_is_translated_to_ttnn():
    compute_kernel = KernelSpec(
        name="compute_kernel",
        thread_type=ThreadType.COMPUTE,
        source="void kernel_main() {}",
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
        unpack_to_dest_fp32=[3, 17],
    )

    config = _get_compute_config(compute_kernel)

    assert config.fp32_dest_acc_en
    assert config.dst_full_sync_en
    assert len(config.unpack_to_dest_mode) == 64
    assert config.unpack_to_dest_mode[3] == ttnn.UnpackToDestMode.UnpackToDestFp32
    assert config.unpack_to_dest_mode[17] == ttnn.UnpackToDestMode.UnpackToDestFp32
    assert config.unpack_to_dest_mode[0] == ttnn.UnpackToDestMode.Default
