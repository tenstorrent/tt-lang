# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Accumulation tests for tile_store {acc = true}.

Validates the full accumulation pipeline on hardware:
  zero-init (fill_tile) -> add_binary_tile -> deferred pack_tile

Each test stores an expression with acc=true. The accumulator is
zero-initialized, so the result is 0 + expr = expr. This verifies
the accumulation lowering (DST assignment, sync placement,
TTKernel emission) produces correct results without corrupting data.
"""

import pytest
import torch
from torch import Tensor

from ..base import ME2ETestBase
from ..config import E2EConfig
from ..builder.dtype_utils import torch_dtype_to_mlir_str
from ..builder.thread_builder import generate_layout_attrs
from ..builder.dm_builder import DMThreadBuilder

import ttl.dialects.ttl as ttl


class AccumulateTestBase(ME2ETestBase):
    """
    Base class for accumulation tests.

    Follows the FusedOpTestBase pattern: custom MLIR template with
    ttl.store {acc = true} in the compute thread.
    """

    OP_NAME: str
    ARITY: int
    INPUT_SHAPE = (1, 1)
    INPUT_DTYPE = torch.bfloat16
    INPUT_RANGE = (-1.0, 1.0)

    @pytest.fixture(scope="class")
    def config(self) -> E2EConfig:
        return E2EConfig(grid_shape=self.INPUT_SHAPE, dtype=self.INPUT_DTYPE)

    def torch_reference(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def get_mlir_template(self, config: E2EConfig) -> str:
        raise NotImplementedError

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self) -> None:
        super().test_compile_to_ttkernel()

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        super().test_translate_to_cpp()

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        super().test_validate_golden()

    @pytest.mark.order(1)
    def test_build_module(self, config: E2EConfig) -> None:
        lo, hi = self.INPUT_RANGE
        torch_inputs = []
        for _ in range(self.ARITY):
            t = torch.rand(config.tensor_shape, dtype=config.dtype) * (hi - lo) + lo
            torch_inputs.append(t)

        from ttl.ir import Context, Module

        mlir_str = self.get_mlir_template(config)

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)
        with ctx:
            module = Module.parse(mlir_str, ctx)
            module.operation.verify()

        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        torch.save(torch_inputs, self.output_file("inputs.pt"))
        golden = self.torch_reference(*torch_inputs)
        torch.save(golden, self.output_file("golden.pt"))


def _tt(rows, cols, dtype):
    """Tile tensor type string."""
    return f"tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>"


def _cb(rows, cols, dtype, bf):
    """CB type string."""
    return f"!ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>"


class TestAccAddBinary(AccumulateTestBase):
    """
    Accumulation of a + b.

    Pattern: y.store(a + b, acc=True)
    Expected: 0 + (a + b) = a + b

    Validates that DST accumulation (fill_tile -> add_binary_tile ->
    pack_tile) produces the same result as a direct add.
    """

    OP_NAME = "acc_add"
    ARITY = 2
    PCC_THRESHOLD = 0.999

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b

    def get_mlir_template(self, config: E2EConfig) -> str:
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor
        tt = _tt(rows, cols, dtype)
        cb = _cb(rows, cols, dtype, bf)

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)
        reader_mlir = dm_builder.build_reader(num_inputs=2)
        writer_mlir = dm_builder.build_writer(output_cbs=[2])

        compute_mlir = f"""
// Compute thread: accumulate a + b.
func.func @compute_acc_add() attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : {cb}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : {cb}
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : {cb}

  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %a_attached = ttl.attach_cb %a, %cb0 : ({tt}, {cb}) -> {tt}
  %b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %b_attached = ttl.attach_cb %b, %cb1 : ({tt}, {cb}) -> {tt}

  %out_reserved = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}

  %sum = ttl.add %a_attached, %b_attached : {tt}, {tt} -> {tt}
  ttl.store %sum, %out_reserved {{acc = true}} : {tt}, {tt}

  ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  return
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


class TestAccExpUnary(AccumulateTestBase):
    """
    Accumulation of exp(a).

    Pattern: y.store(exp(a), acc=True)
    Expected: 0 + exp(a) = exp(a)

    Validates accumulation with a unary op chain.
    """

    OP_NAME = "acc_exp"
    ARITY = 1
    INPUT_RANGE = (-2.0, 2.0)
    PCC_THRESHOLD = 0.999

    def torch_reference(self, a: Tensor) -> Tensor:
        return torch.exp(a)

    def get_mlir_template(self, config: E2EConfig) -> str:
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor
        tt = _tt(rows, cols, dtype)
        cb = _cb(rows, cols, dtype, bf)

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)
        reader_mlir = dm_builder.build_reader(num_inputs=1)
        writer_mlir = dm_builder.build_writer(output_cbs=[1])

        compute_mlir = f"""
// Compute thread: accumulate exp(a).
func.func @compute_acc_exp() attributes {{ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : {cb}
  %cb_out = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : {cb}

  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %a_attached = ttl.attach_cb %a, %cb0 : ({tt}, {cb}) -> {tt}

  %out_reserved = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}

  %result = ttl.exp %a_attached : {tt} -> {tt}
  ttl.store %result, %out_reserved {{acc = true}} : {tt}, {tt}

  ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  return
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


class TestAccPassthrough(AccumulateTestBase):
    """
    Accumulation passthrough: store input directly with acc=true.

    Pattern: y.store(a, acc=True)
    Expected: 0 + a = a

    Tests passthrough lowering (LowerStoreToCompute) with accumulation.
    The compute body contains only a copy_tile + acc tile_store.
    """

    OP_NAME = "acc_passthrough"
    ARITY = 1
    PCC_THRESHOLD = 0.999

    def torch_reference(self, a: Tensor) -> Tensor:
        return a

    def get_mlir_template(self, config: E2EConfig) -> str:
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor
        tt = _tt(rows, cols, dtype)
        cb = _cb(rows, cols, dtype, bf)

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)
        reader_mlir = dm_builder.build_reader(num_inputs=1)
        writer_mlir = dm_builder.build_writer(output_cbs=[1])

        compute_mlir = f"""
// Compute thread: accumulate passthrough (identity store).
func.func @compute_acc_passthrough() attributes {{ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : {cb}
  %cb_out = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : {cb}

  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %a_attached = ttl.attach_cb %a, %cb0 : ({tt}, {cb}) -> {tt}

  %out_reserved = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}

  ttl.store %a_attached, %out_reserved {{acc = true}} : {tt}, {tt}

  ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  return
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


class TestAccMultiStore(AccumulateTestBase):
    """
    Multi-store accumulation: two passthrough acc stores to the same view.

    Pattern: y.store(a, acc=True); y.store(b, acc=True)
    Expected: 0 + a + b = a + b

    MLIR uses a pre-formed ttl.compute, bypassing ConvertTTLToCompute
    which cannot yet fuse multiple stores into one compute body.
    """

    OP_NAME = "acc_multi_store"
    ARITY = 2
    INPUT_RANGE = (0.5, 1.5)

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b

    def get_mlir_template(self, config: E2EConfig) -> str:
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor
        tt = _tt(rows, cols, dtype)
        cb = _cb(rows, cols, dtype, bf)
        tile = f"!ttcore.tile<32x32, {dtype}>"

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)
        reader_mlir = dm_builder.build_reader(num_inputs=2)
        writer_mlir = dm_builder.build_writer(output_cbs=[2])

        compute_mlir = f"""
// Compute thread: multi-store accumulation (a + b via two acc stores).
func.func @compute_acc_multi_store() attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : {cb}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : {cb}
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : {cb}

  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %a_attached = ttl.attach_cb %a, %cb0 : ({tt}, {cb}) -> {tt}
  %b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}
  %b_attached = ttl.attach_cb %b, %cb1 : ({tt}, {cb}) -> {tt}

  %out_view = ttl.cb_reserve %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> {tt}

  ttl.store %a_attached, %out_view {{acc = true}} : {tt}, {tt}
  ttl.store %b_attached, %out_view {{acc = true}} : {tt}, {tt}

  ttl.cb_push %cb_out : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  ttl.cb_pop %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  return
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""
