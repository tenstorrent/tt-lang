# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-free compute correctness through the normal device launcher."""

import importlib.util
import re

import pytest
import torch
import ttl
from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device


@pytest.fixture(autouse=True)
def forbid_metal_descriptors(request, monkeypatch):
    if request.node.callspec.params.get("memory_model", "compiler-l1") != "compiler-l1":
        return
    import ttl.kernel_runner as runner

    def reject_descriptors(*args, **kwargs):
        pytest.fail("compiler-l1 constructed Metal DFB descriptors")

    monkeypatch.setattr(runner, "build_cb_descriptors", reject_descriptors)


def _make_binary(*, multiply=False, subtract=False):
    assert not (multiply and subtract)

    @ttl.operation(grid=(1, 1))
    def binary(lhs, rhs, output):
        lhs_storage = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=3)
        rhs_storage = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=3)
        output_storage = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=3
        )

        @ttl.compute()
        def compute():
            for iteration in range(7):
                with (
                    lhs_storage.wait() as lhs_block,
                    rhs_storage.wait() as rhs_block,
                    output_storage.reserve() as output_block,
                ):
                    if multiply:
                        output_block.store(lhs_block * rhs_block)
                    elif subtract:
                        output_block.store(lhs_block - rhs_block)
                    else:
                        output_block.store(lhs_block + rhs_block)

        @ttl.datamovement()
        def reader():
            for iteration in range(7):
                with lhs_storage.reserve() as lhs_block:
                    ttl.copy(lhs[iteration : iteration + 1, 0:1], lhs_block).wait()
                with rhs_storage.reserve() as rhs_block:
                    ttl.copy(rhs[iteration : iteration + 1, 0:1], rhs_block).wait()

        @ttl.datamovement()
        def writer():
            for iteration in range(7):
                with output_storage.wait() as output_block:
                    ttl.copy(
                        output_block, output[iteration : iteration + 1, 0:1]
                    ).wait()

    return binary


@pytest.mark.parametrize("fpu", [True, False], ids=["fpu", "sfpu"])
@pytest.mark.parametrize("operation_name", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_binary(device, dtype, allocator, memory_model, operation_name, fpu):
    operation = _make_binary(
        multiply=operation_name == "multiply",
        subtract=operation_name == "subtract",
    )

    for invocation in range(3):
        # Exact TF32 inputs isolate storage corruption from FPU input rounding.
        lhs_reference = torch.randint(-64, 65, (224, 32)).to(dtype) / 32
        rhs_reference = torch.randint(-64, 65, (224, 32)).to(dtype) / 32
        if operation_name == "multiply":
            # Power-of-two factors also avoid fidelity-dependent product rounding.
            rhs_reference = (2.0 ** torch.randint(-2, 3, (224, 32))).to(dtype)
        if operation_name == "multiply":
            expected = lhs_reference * rhs_reference
        elif operation_name == "subtract":
            expected = lhs_reference - rhs_reference
        else:
            expected = lhs_reference + rhs_reference
        lhs = allocator(lhs_reference, device)
        rhs = allocator(rhs_reference, device)
        output = allocator(torch.zeros_like(expected), device)
        options = f"--ttl-memory-model={memory_model}"
        if not fpu:
            options += " --no-ttl-fpu-binary-ops"
        operation(lhs, rhs, output, options=options)
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


def _load_operation(tmp_path, name, source):
    source_file = tmp_path / f"{name}.py"
    source_file.write_text(source)
    spec = importlib.util.spec_from_file_location(name, source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _make_above_descriptor_limit_add(tmp_path, input_count):
    lines = [
        "import ttl",
        "@ttl.operation(grid=(1, 1))",
        "def above_descriptor_limit_add(source, destination):",
    ]
    for region in range(input_count):
        lines.append(
            f"    storage_{region} = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=1)"
        )
    lines += [
        "    output_storage = ttl.make_dataflow_buffer_like(destination, shape=(1, 1), block_count=1)",
        "    @ttl.compute()",
        "    def compute():",
    ]
    lines += [
        "        with storage_0.wait() as lhs, storage_1.wait() as rhs:",
        "            with output_storage.reserve() as output:",
        "                output.store(lhs + rhs)",
    ]
    lines += ["    @ttl.datamovement()", "    def reader():"]
    for region in range(input_count):
        lines += [
            f"        with storage_{region}.reserve() as block:",
            f"            ttl.copy(source[{region}:{region + 1}, 0:1], block).wait()",
        ]
    lines += [
        "    @ttl.datamovement()",
        "    def writer():",
        "        with output_storage.wait() as block:",
        "            ttl.copy(block, destination[0:1, 0:1]).wait()",
    ]
    for region in range(2, input_count):
        lines += [
            f"        with storage_{region}.wait() as block:",
            f"            ttl.copy(block, destination[{region - 1}:{region}, 0:1]).wait()",
        ]
    return _load_operation(
        tmp_path, "above_descriptor_limit_add", "\n".join(lines) + "\n"
    )


# Proves that arithmetic remains usable after logical DFB count exceeds 64.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_l1_compute_above_descriptor_limit(device, dtype, tmp_path, monkeypatch):
    input_count = 65
    operation = _make_above_descriptor_limit_add(tmp_path, input_count)
    final_ir = tmp_path / "final.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_ir))
    options = "--ttl-memory-model=compiler-l1"
    for invocation in range(2):
        reference = torch.randint(-64, 65, (input_count, 32, 32)).to(dtype) / 32
        expected = torch.cat((reference[0:1] + reference[1:2], reference[2:]))
        source = to_dram(reference.reshape(-1, 32), device)
        destination = to_dram(torch.zeros_like(expected.reshape(-1, 32)), device)
        operation(source, destination, options=options)
        assert_allclose(
            ttnn.to_torch(destination).float(),
            expected.reshape(-1, 32).float(),
            rtol=0,
            atol=0,
        )
    offsets = re.findall(r"l1_payload_offset = (\d+)", final_ir.read_text())
    assert len(offsets) == input_count + 1
    assert len(set(offsets)) == input_count + 1


def _make_matmul(rows, inner, columns):
    @ttl.operation(grid=(1, 1))
    def matmul(lhs, rhs, output):
        lhs_storage = ttl.make_dataflow_buffer_like(
            lhs, shape=(rows, inner), block_count=2
        )
        rhs_storage = ttl.make_dataflow_buffer_like(
            rhs, shape=(inner, columns), block_count=2
        )
        output_storage = ttl.make_dataflow_buffer_like(
            output, shape=(rows, columns), block_count=2
        )

        @ttl.compute()
        def compute():
            with (
                lhs_storage.wait() as lhs_block,
                rhs_storage.wait() as rhs_block,
                output_storage.reserve() as output_block,
            ):
                output_block.store(lhs_block @ rhs_block)

        @ttl.datamovement()
        def reader():
            with lhs_storage.reserve() as lhs_block:
                ttl.copy(lhs[0:rows, 0:inner], lhs_block).wait()
            with rhs_storage.reserve() as rhs_block:
                ttl.copy(rhs[0:inner, 0:columns], rhs_block).wait()

        @ttl.datamovement()
        def writer():
            with output_storage.wait() as output_block:
                ttl.copy(output_block, output[0:rows, 0:columns]).wait()

    return matmul


@pytest.mark.parametrize(
    "dimensions", [(1, 1, 1), (2, 2, 2), (1, 3, 2), (1, 1, 5), (3, 2, 3)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
def test_l1_matmul(device, dimensions, dtype, memory_model, allocator):

    rows, inner, columns = dimensions
    operation = _make_matmul(rows, inner, columns)
    for invocation in range(3):
        lhs_reference = (2 * torch.randint(0, 2, (rows * 32, inner * 32)) - 1).to(dtype)
        rhs_reference = (2 * torch.randint(0, 2, (inner * 32, columns * 32)) - 1).to(
            dtype
        )
        expected = lhs_reference @ rhs_reference
        lhs = allocator(lhs_reference, device)
        rhs = allocator(rhs_reference, device)
        output = allocator(torch.zeros_like(expected), device)
        operation(lhs, rhs, output, options=f"--ttl-memory-model={memory_model}")
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


@ttl.operation(grid=(1, 1))
def l1_residual_chain(lhs, rhs, output):
    lhs_storage = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=3)
    rhs_storage = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=3)
    sum_storage = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    product_storage = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=3)

    @ttl.compute()
    def compute():
        for iteration in range(7):
            with lhs_storage.wait() as residual, rhs_storage.wait() as rhs_block:
                with sum_storage.reserve() as sum_block:
                    sum_block.store(residual + rhs_block)
                with (
                    sum_storage.wait() as sum_block,
                    product_storage.reserve() as product_block,
                ):
                    product_block.store(sum_block * rhs_block)
                with (
                    product_storage.wait() as product_block,
                    output_storage.reserve() as output_block,
                ):
                    output_block.store(product_block + residual)

    @ttl.datamovement()
    def reader():
        for iteration in range(7):
            with lhs_storage.reserve() as block:
                ttl.copy(lhs[iteration : iteration + 1, 0:1], block).wait()
            with rhs_storage.reserve() as block:
                ttl.copy(rhs[iteration : iteration + 1, 0:1], block).wait()

    @ttl.datamovement()
    def writer():
        for iteration in range(7):
            with output_storage.wait() as block:
                ttl.copy(block, output[iteration : iteration + 1, 0:1]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
def test_l1_residual_chain(device, dtype, memory_model, reuse, allocator):

    options = f"--ttl-memory-model={memory_model}"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    for invocation in range(3):
        lhs_reference = torch.randint(-8, 9, (224, 32)).to(dtype) / 8
        rhs_reference = (2.0 ** torch.randint(-1, 2, (224, 32))).to(dtype)
        expected = (lhs_reference + rhs_reference) * rhs_reference + lhs_reference
        lhs = allocator(lhs_reference, device)
        rhs = allocator(rhs_reference, device)
        output = allocator(torch.zeros_like(expected), device)
        l1_residual_chain(lhs, rhs, output, options=options)
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


def _make_reduce(tmp_path, dimensions, tiles, maximum):
    rows = tiles if 0 in dimensions else 1
    columns = tiles if 1 in dimensions else 1
    reduce_name = "reduce_max" if maximum else "reduce_sum"
    dimension_list = ", ".join(str(dimension) for dimension in dimensions)
    source = f"""import ttl
import torch
@ttl.operation(grid=(1, 1))
def reduction(source, output):
    input_storage = ttl.make_dataflow_buffer_like(source, shape=({rows}, {columns}), block_count=2)
    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=2)
    @ttl.compute()
    def compute():
        with input_storage.wait() as input_block, output_storage.reserve() as output_block:
            output_block.store(ttl.math.{reduce_name}(input_block, dims=[{dimension_list}], shape=(1, 1)))
    @ttl.datamovement()
    def reader():
        with input_storage.reserve() as block:
            ttl.copy(source[0:{rows}, 0:{columns}], block).wait()
    @ttl.datamovement()
    def writer():
        with output_storage.wait() as block:
            ttl.copy(block, output[0:1, 0:1]).wait()
"""
    return _load_operation(tmp_path, "reduction", source)


@pytest.mark.parametrize(
    "dimensions", [(0,), (1,), (0, 1)], ids=["columns", "rows", "scalar"]
)
@pytest.mark.parametrize("tiles", [1, 3], ids=["one_tile", "three_tiles"])
@pytest.mark.parametrize("maximum", [False, True], ids=["sum", "max"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
def test_l1_reduce(
    device, dimensions, tiles, maximum, dtype, memory_model, tmp_path, allocator
):
    operation = _make_reduce(tmp_path, dimensions, tiles, maximum)
    rows = tiles * 32 if 0 in dimensions else 32
    columns = tiles * 32 if 1 in dimensions else 32
    for invocation in range(3):
        reference = (2 * torch.randint(0, 2, (rows, columns)) - 1).to(dtype)
        if maximum:
            reference = torch.randint(-16, 17, (rows, columns)).to(dtype)
        expected = (
            reference.amax(dim=dimensions, keepdim=True)
            if maximum
            else reference.sum(dim=dimensions, keepdim=True)
        )
        source = allocator(reference, device)
        output = allocator(torch.zeros((32, 32), dtype=dtype), device)
        operation(source, output, options=f"--ttl-memory-model={memory_model}")
        actual = ttnn.to_torch(output).float()
        if dimensions == (0,):
            actual = actual[:1, :]
        elif dimensions == (1,):
            actual = actual[:, :1]
        else:
            actual = actual[:1, :1]
        assert_allclose(actual, expected.float(), rtol=0, atol=0)


def _make_unary(tmp_path, expression):
    source = f"""import ttl
import torch
@ttl.operation(grid=(1, 1))
def unary(source, output):
    input_storage = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=3)
    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=3)
    @ttl.compute()
    def compute():
        for iteration in range(7):
            with input_storage.wait() as input_block, output_storage.reserve() as output_block:
                output_block.store({expression})
    @ttl.datamovement()
    def reader():
        for iteration in range(7):
            with input_storage.reserve() as block:
                ttl.copy(source[iteration:iteration + 1, 0:1], block).wait()
    @ttl.datamovement()
    def writer():
        for iteration in range(7):
            with output_storage.wait() as block:
                ttl.copy(block, output[iteration:iteration + 1, 0:1]).wait()
"""
    return _load_operation(tmp_path, "unary", source)


@pytest.mark.parametrize(
    "axes", [(0,), (1,), (0, 1)], ids=["rows", "columns", "scalar"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_broadcast(device, axes, dtype, allocator, memory_model, tmp_path):
    operation = _make_unary(
        tmp_path, f"ttl.block.broadcast(input_block, dims={list(axes)}, shape=(1, 1))"
    )
    for invocation in range(3):
        reference = torch.randn((7, 32, 32)).to(dtype)
        # Default FP32 broadcast uses the TF32 source-register format.
        expected = reference
        if dtype == torch.float32:
            expected = (reference.view(torch.int32) & -8192).view(torch.float32)
        if 0 in axes:
            expected = expected[:, :1, :]
        if 1 in axes:
            expected = expected[:, :, :1]
        expected = expected.expand_as(reference).reshape(224, 32)
        source = allocator(reference.reshape(224, 32), device)
        output = allocator(torch.zeros_like(expected), device)
        operation(source, output, options=f"--ttl-memory-model={memory_model}")
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("activation", ["rsqrt", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_activation(device, activation, dtype, allocator, memory_model, tmp_path):
    operation = _make_unary(tmp_path, f"ttl.math.{activation}(input_block)")
    for invocation in range(3):
        reference = torch.randn((224, 32)).to(dtype)
        if activation == "rsqrt":
            reference = reference.abs() + 0.25
        expected = getattr(torch, activation)(reference.float())
        source = allocator(reference, device)
        output = allocator(torch.zeros_like(reference), device)
        operation(source, output, options=f"--ttl-memory-model={memory_model}")
        tolerance = 0.01 if dtype == torch.bfloat16 else 0.002
        assert_allclose(
            ttnn.to_torch(output).float(), expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
def test_l1_rms_normalization(device, dtype, memory_model, tmp_path, allocator):
    operation = _make_unary(
        tmp_path,
        f"input_block * ttl.block.broadcast(ttl.math.rsqrt(ttl.math.reduce_sum(input_block * input_block, dims=[1], shape=(1, 1)) * 0.03125 + ttl.block.fill(0.00001, shape=(1, 1), dtype={dtype})), dims=[1], shape=(1, 1))",
    )
    for invocation in range(3):
        reference = (torch.randn((224, 32)) * 0.5).to(dtype)
        expected = reference.float() * torch.rsqrt(
            reference.float().square().mean(dim=1, keepdim=True) + 0.00001
        )
        source = allocator(reference, device)
        output = allocator(torch.zeros_like(reference), device)
        operation(source, output, options=f"--ttl-memory-model={memory_model}")
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.004
        assert_allclose(
            ttnn.to_torch(output).float(), expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("multiply", [False, True], ids=["add", "multiply"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_sfpu_precision(device, multiply, dtype, allocator, memory_model):
    # Full-precision inputs expose accidental TF32 conversion during direct unpack.
    operation = _make_binary(multiply=multiply)
    for invocation in range(3):
        lhs_reference = torch.randn((224, 32)).to(dtype)
        rhs_reference = torch.randn((224, 32)).to(dtype)
        expected = (
            lhs_reference * rhs_reference if multiply else lhs_reference + rhs_reference
        )
        lhs = allocator(lhs_reference, device)
        rhs = allocator(rhs_reference, device)
        output = allocator(torch.zeros_like(expected), device)
        operation(
            lhs,
            rhs,
            output,
            options=f"--ttl-memory-model={memory_model} --no-ttl-fpu-binary-ops",
        )
        tolerance = 0.008 if dtype == torch.bfloat16 else 0.000001
        assert_allclose(
            ttnn.to_torch(output).float(),
            expected.float(),
            rtol=tolerance,
            atol=tolerance,
        )


def _make_kimi_situ_mlp_residual(normalize):
    @ttl.operation(grid=(1, 1))
    def kimi_situ_mlp_residual(source, gate_weight, up_weight, down_weight, output):
        input_storage = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=2
        )
        residual_storage = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=2
        )
        normalized_storage = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=2
        )
        gate_weight_storage = ttl.make_dataflow_buffer_like(
            gate_weight, shape=(1, 1), block_count=1
        )
        up_weight_storage = ttl.make_dataflow_buffer_like(
            up_weight, shape=(1, 1), block_count=1
        )
        down_weight_storage = ttl.make_dataflow_buffer_like(
            down_weight, shape=(1, 1), block_count=1
        )
        gate_storage = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=2
        )
        up_storage = ttl.make_dataflow_buffer_like(source, shape=(1, 1), block_count=2)
        activation_storage = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=2
        )
        output_storage = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=2
        )

        @ttl.compute()
        def compute():
            with (
                gate_weight_storage.wait() as gate_weights,
                up_weight_storage.wait() as up_weights,
                down_weight_storage.wait() as down_weights,
            ):
                for iteration in range(5):
                    with (
                        input_storage.wait() as input_block,
                        normalized_storage.reserve() as normalized_block,
                        residual_storage.reserve() as residual_block,
                    ):
                        residual_block.store(input_block)
                        if normalize:
                            squared = input_block * input_block
                            total = ttl.math.reduce_sum(squared, dims=[1], shape=(1, 1))
                            biased = total * 0.03125 + ttl.block.fill(
                                0.00001, shape=(1, 1), dtype=input_block.dtype
                            )
                            inverse = ttl.math.rsqrt(biased)
                            normalized_block.store(
                                input_block
                                * ttl.block.broadcast(inverse, dims=[1], shape=(1, 1))
                            )
                        else:
                            normalized_block.store(input_block)
                    with normalized_storage.wait() as input_block:
                        with gate_storage.reserve() as gate_block:
                            gate_block.store(input_block @ gate_weights)
                        with up_storage.reserve() as up_block:
                            up_block.store(input_block @ up_weights)
                    with (
                        gate_storage.wait() as gate_block,
                        up_storage.wait() as up_block,
                        activation_storage.reserve() as activation_block,
                    ):
                        limited_gate = 4.0 * ttl.math.tanh(gate_block * 0.25)
                        limited_up = 25.0 * ttl.math.tanh(up_block * 0.04)
                        activation_block.store(
                            limited_gate * ttl.math.sigmoid(gate_block) * limited_up
                        )
                    with (
                        activation_storage.wait() as activation_block,
                        residual_storage.wait() as residual_block,
                        output_storage.reserve() as output_block,
                    ):
                        output_block.store(
                            activation_block @ down_weights + residual_block
                        )

        @ttl.datamovement()
        def reader():
            with gate_weight_storage.reserve() as block:
                ttl.copy(gate_weight[0:1, 0:1], block).wait()
            with up_weight_storage.reserve() as block:
                ttl.copy(up_weight[0:1, 0:1], block).wait()
            with down_weight_storage.reserve() as block:
                ttl.copy(down_weight[0:1, 0:1], block).wait()
            for iteration in range(5):
                with input_storage.reserve() as block:
                    ttl.copy(source[iteration : iteration + 1, 0:1], block).wait()

        @ttl.datamovement()
        def writer():
            for iteration in range(5):
                with output_storage.wait() as block:
                    ttl.copy(block, output[iteration : iteration + 1, 0:1]).wait()

    return kimi_situ_mlp_residual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
@pytest.mark.parametrize(
    "normalize", [False, True], ids=["projection", "normalized_projection"]
)
def test_l1_kimi_situ_mlp_residual(
    device, dtype, allocator, reuse, memory_model, normalize
):
    operation = _make_kimi_situ_mlp_residual(normalize)
    options = f"--ttl-memory-model={memory_model}"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    for invocation in range(3):
        reference = (torch.randn((160, 32)) * 0.25).to(dtype)
        weights = [(torch.randn((32, 32)) * 0.125).to(dtype) for weight in range(3)]
        projected_input = reference.float()
        if normalize:
            projected_input = (
                (
                    projected_input
                    * torch.rsqrt(
                        projected_input.square().mean(dim=1, keepdim=True) + 0.00001
                    )
                )
                .to(dtype)
                .float()
            )
        gate = (projected_input @ weights[0].float()).to(dtype).float()
        up = (projected_input @ weights[1].float()).to(dtype).float()
        activated = (
            (
                4.0
                * torch.tanh(gate * 0.25)
                * torch.sigmoid(gate)
                * 25.0
                * torch.tanh(up * 0.04)
            )
            .to(dtype)
            .float()
        )
        expected = (
            (activated @ weights[2].float() + reference.float()).to(dtype).float()
        )
        inputs = [allocator(reference, device)] + [
            allocator(weight, device) for weight in weights
        ]
        output = allocator(torch.zeros_like(reference), device)
        operation(*inputs, output, options=options)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.004
        assert_allclose(
            ttnn.to_torch(output).float(), expected, rtol=tolerance, atol=tolerance
        )


def _make_dependent_state(tmp_path, steps):
    lines = [
        "import ttl",
        "@ttl.operation(grid=(1, 1))",
        "def dependent_state(seed, updates, output):",
    ]
    for state in range(steps + 1):
        lines.append(
            f"    state_{state} = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=1)"
        )
    lines += [
        "    update_storage = ttl.make_dataflow_buffer_like(updates, shape=(1, 1), block_count=3)",
        "    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=3)",
        "    @ttl.compute()",
        "    def compute():",
    ]
    for step in range(steps):
        lines += [
            f"        with state_{step}.wait() as previous, update_storage.wait() as update, state_{step + 1}.reserve() as following, output_storage.reserve() as result:",
            "            next_value = previous * 0.5 + update",
            "            following.store(next_value)",
            "            result.store(next_value + previous)",
        ]
    lines += [
        f"        final_state = state_{steps}.wait()",
        "        final_state.pop()",
        "    @ttl.datamovement()",
        "    def reader():",
        "        with state_0.reserve() as block:",
        "            ttl.copy(seed[0:1, 0:1], block).wait()",
        f"        for iteration in range({steps}):",
        "            with update_storage.reserve() as block:",
        "                ttl.copy(updates[iteration:iteration + 1, 0:1], block).wait()",
        "    @ttl.datamovement()",
        "    def writer():",
        f"        for iteration in range({steps}):",
        "            with output_storage.wait() as block:",
        "                ttl.copy(block, output[iteration:iteration + 1, 0:1]).wait()",
    ]
    return _load_operation(tmp_path, "dependent_state", "\n".join(lines) + "\n")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_dependent_state(device, dtype, allocator, reuse, memory_model, tmp_path):
    steps = 5
    operation = _make_dependent_state(tmp_path, steps)
    options = f"--ttl-memory-model={memory_model}"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    for invocation in range(3):
        # Five half-decay updates stay exactly representable in both dtypes.
        seed_reference = torch.randint(-1, 2, (32, 32)).to(dtype)
        updates_reference = torch.randint(-1, 2, (steps, 32, 32)).to(dtype)
        previous = seed_reference
        results = []
        for update in updates_reference:
            following = previous * 0.5 + update
            results.append(following + previous)
            previous = following
        expected = torch.cat(results)
        seed = allocator(seed_reference, device)
        updates = allocator(updates_reference.reshape(-1, 32), device)
        output = allocator(torch.zeros_like(expected), device)
        operation(seed, updates, output, options=options)
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_transpose(device, dtype, allocator, memory_model, tmp_path):
    operation = _make_unary(tmp_path, "ttl.math.transpose(input_block)")
    for invocation in range(3):
        reference = torch.randn((7, 32, 32)).to(dtype)
        expected = reference.transpose(1, 2).reshape(224, 32)
        if dtype == torch.float32:
            expected = (expected.view(torch.int32) & -8192).view(torch.float32)
        source = allocator(reference.reshape(224, 32), device)
        output = allocator(torch.zeros_like(expected), device)
        operation(source, output, options=f"--ttl-memory-model={memory_model}")
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)


@ttl.operation(grid=(1, 1))
def l1_attention(query, key, value, output):
    query_storage = ttl.make_dataflow_buffer_like(query, shape=(1, 1), block_count=1)
    key_storage = ttl.make_dataflow_buffer_like(key, shape=(1, 1), block_count=1)
    value_storage = ttl.make_dataflow_buffer_like(value, shape=(1, 1), block_count=1)
    score_storage = ttl.make_dataflow_buffer_like(query, shape=(1, 1), block_count=1)
    score_elementwise_storage = ttl.make_dataflow_buffer_like(
        query, shape=(1, 1), block_count=1
    )
    numerator_elementwise_storage = ttl.make_dataflow_buffer_like(
        query, shape=(1, 1), block_count=1
    )
    shifted_storage = ttl.make_dataflow_buffer_like(query, shape=(1, 1), block_count=1)
    numerator_storage = ttl.make_dataflow_buffer_like(
        query, shape=(1, 1), block_count=1
    )
    probability_storage = ttl.make_dataflow_buffer_like(
        query, shape=(1, 1), block_count=1
    )
    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        with (
            query_storage.wait() as query_block,
            key_storage.wait() as key_block,
            score_storage.reserve() as score_block,
            score_elementwise_storage.reserve() as elementwise_score,
        ):
            scores = (query_block @ ttl.math.transpose(key_block)) * 0.125
            score_block.store(scores)
            elementwise_score.store(scores)
        # Reduction and SFPU consumers require separate FP32 unpack configurations.
        with score_storage.wait() as score_block:
            maximum = ttl.math.reduce_max(score_block, dims=[1], shape=(1, 1))
            with (
                score_elementwise_storage.wait() as copied_score,
                shifted_storage.reserve() as shifted_block,
            ):
                shifted_block.store(
                    copied_score - ttl.block.broadcast(maximum, dims=[1], shape=(1, 1))
                )
        with (
            shifted_storage.wait() as shifted_block,
            numerator_storage.reserve() as numerator_block,
            numerator_elementwise_storage.reserve() as elementwise_numerator,
        ):
            numerator = ttl.math.exp(shifted_block)
            numerator_block.store(numerator)
            elementwise_numerator.store(numerator)
        with numerator_storage.wait() as numerator_block:
            denominator = ttl.math.reduce_sum(numerator_block, dims=[1], shape=(1, 1))
            inverse = ttl.math.recip(denominator)
            with (
                numerator_elementwise_storage.wait() as copied_numerator,
                probability_storage.reserve() as probability_block,
            ):
                probability_block.store(
                    copied_numerator
                    * ttl.block.broadcast(inverse, dims=[1], shape=(1, 1))
                )
        with (
            probability_storage.wait() as probability_block,
            value_storage.wait() as value_block,
            output_storage.reserve() as output_block,
        ):
            output_block.store(probability_block @ value_block)

    @ttl.datamovement()
    def reader():
        with query_storage.reserve() as block:
            ttl.copy(query[0:1, 0:1], block).wait()
        with key_storage.reserve() as block:
            ttl.copy(key[0:1, 0:1], block).wait()
        with value_storage.reserve() as block:
            ttl.copy(value[0:1, 0:1], block).wait()

    @ttl.datamovement()
    def writer():
        with output_storage.wait() as block:
            ttl.copy(block, output[0:1, 0:1]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_attention(device, dtype, allocator, reuse, memory_model):
    options = f"--ttl-memory-model={memory_model}"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    for invocation in range(3):
        references = [(torch.randn((32, 32)) * 0.5).to(dtype) for operand in range(3)]
        query_reference, key_reference, value_reference = references
        scores = (
            ((query_reference.float() @ key_reference.float().T) * 0.125)
            .to(dtype)
            .float()
        )
        probabilities = scores.softmax(dim=1).to(dtype).float()
        expected = (probabilities @ value_reference.float()).to(dtype).float()
        inputs = [allocator(reference, device) for reference in references]
        output = allocator(torch.zeros_like(query_reference), device)
        l1_attention(*inputs, output, options=options)
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.003
        assert_allclose(
            ttnn.to_torch(output).float(), expected, rtol=tolerance, atol=tolerance
        )


@ttl.operation(grid=(1, 1))
def l1_expert_merge(partials, routing, residual, output):
    first_storage = ttl.make_dataflow_buffer_like(partials, shape=(1, 1), block_count=2)
    second_storage = ttl.make_dataflow_buffer_like(
        partials, shape=(1, 1), block_count=2
    )
    first_weight_storage = ttl.make_dataflow_buffer_like(
        routing, shape=(1, 1), block_count=2
    )
    second_weight_storage = ttl.make_dataflow_buffer_like(
        routing, shape=(1, 1), block_count=2
    )
    residual_storage = ttl.make_dataflow_buffer_like(
        residual, shape=(1, 1), block_count=2
    )
    output_storage = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for iteration in range(5):
            with (
                first_storage.wait() as first,
                second_storage.wait() as second,
                first_weight_storage.wait() as first_weight,
                second_weight_storage.wait() as second_weight,
                residual_storage.wait() as retained,
                output_storage.reserve() as result,
            ):
                first_scale = ttl.block.broadcast(
                    first_weight, dims=[0, 1], shape=(1, 1)
                )
                second_scale = ttl.block.broadcast(
                    second_weight, dims=[0, 1], shape=(1, 1)
                )
                result.store(first * first_scale + second * second_scale + retained)

    @ttl.datamovement()
    def reader():
        for iteration in range(5):
            with first_storage.reserve() as block:
                ttl.copy(partials[2 * iteration : 2 * iteration + 1, 0:1], block).wait()
            with second_storage.reserve() as block:
                ttl.copy(
                    partials[2 * iteration + 1 : 2 * iteration + 2, 0:1], block
                ).wait()
            with first_weight_storage.reserve() as block:
                ttl.copy(routing[2 * iteration : 2 * iteration + 1, 0:1], block).wait()
            with second_weight_storage.reserve() as block:
                ttl.copy(
                    routing[2 * iteration + 1 : 2 * iteration + 2, 0:1], block
                ).wait()
            with residual_storage.reserve() as block:
                ttl.copy(residual[iteration : iteration + 1, 0:1], block).wait()

    @ttl.datamovement()
    def writer():
        for iteration in range(5):
            with output_storage.wait() as block:
                ttl.copy(block, output[iteration : iteration + 1, 0:1]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("allocator", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("reuse", [False, True], ids=["distinct", "reuse"])
@pytest.mark.parametrize("memory_model", ["metal-cb", "compiler-l1"])
def test_l1_expert_merge(device, dtype, allocator, reuse, memory_model):
    options = f"--ttl-memory-model={memory_model}"
    if not reuse:
        options += " --no-ttl-reuse-user-dfbs"
    for invocation in range(3):
        partial_reference = torch.randint(-8, 9, (10, 32, 32)).to(dtype) / 8
        residual_reference = torch.randint(-8, 9, (5, 32, 32)).to(dtype) / 8
        weights = torch.randint(1, 4, (5,)).to(dtype) / 4
        routing_reference = torch.zeros((10, 32, 32), dtype=dtype)
        routing_reference[0::2, 0, 0] = weights
        routing_reference[1::2, 0, 0] = 1 - weights
        expected = (
            partial_reference[0::2] * weights[:, None, None]
            + partial_reference[1::2] * (1 - weights[:, None, None])
            + residual_reference
        ).reshape(160, 32)
        partials = allocator(partial_reference.reshape(320, 32), device)
        routing = allocator(routing_reference.reshape(320, 32), device)
        residual = allocator(residual_reference.reshape(160, 32), device)
        output = allocator(torch.zeros_like(expected), device)
        l1_expert_merge(partials, routing, residual, output, options=options)
        assert_allclose(ttnn.to_torch(output).float(), expected.float(), rtol=0, atol=0)
