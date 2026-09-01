# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TT device options used by the TTL Python wrapper."""

from unittest import mock

import pytest

import ttl
import ttl.atom as ttl_atom
import ttl.dialects.ttl as ttl_dialect
import ttl.ttl_api as ttl_api
from ttl import ProgramRuntimeResources
from ttl.constants import SUPPORTED_MATH_FIDELITIES
from ttl.ir import Context, Module


class _TensorWithDevice:
    def __init__(self, device):
        self._device = device

    def device(self):
        return self._device


class _GridSize:
    x = 1
    y = 1


class _BoundingBox:
    @staticmethod
    def grid_size():
        return _GridSize()


class _CoreRanges:
    @staticmethod
    def bounding_box():
        return _BoundingBox()


class _DeviceGrid:
    x = 8
    y = 8


class _DeviceWithMeshShape:
    def __init__(self, shape):
        self.shape = shape

    @staticmethod
    def compute_with_storage_grid_size():
        return _DeviceGrid()


class _DeviceWithArchMethod:
    def __init__(self, arch):
        self._arch = arch

    def arch(self):
        return self._arch


class _DeviceWithArchAttribute:
    def __init__(self, arch):
        self.arch = arch


class _DeviceWithRaisingArch:
    @property
    def arch(self):
        raise RuntimeError("device handle closed")


class TestDeviceTargetArch:
    @pytest.fixture(autouse=True)
    def _patch_tensor_detection(self):
        with mock.patch.object(
            ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _TensorWithDevice)
        ):
            yield

    def test_arch_method(self):
        device = _DeviceWithArchMethod("Arch.BLACKHOLE")
        assert ttl_api._device_target_arch((_TensorWithDevice(device),)) == "blackhole"

    def test_arch_attribute(self):
        device = _DeviceWithArchAttribute("Arch.WORMHOLE_B0")
        assert (
            ttl_api._device_target_arch((_TensorWithDevice(device),)) == "wormhole_b0"
        )

    def test_arch_without_dot_prefix(self):
        device = _DeviceWithArchAttribute("BLACKHOLE")
        assert ttl_api._device_target_arch((_TensorWithDevice(device),)) == "blackhole"

    def test_quasar_arch_is_rejected(self):
        device = _DeviceWithArchMethod("Arch.QUASAR")
        with pytest.raises(ValueError, match="Unsupported TT device architecture"):
            ttl_api._device_target_arch((_TensorWithDevice(device),))

    def test_unknown_arch_is_rejected(self):
        device = _DeviceWithArchAttribute("future_arch")
        with pytest.raises(ValueError, match="Unsupported TT device architecture"):
            ttl_api._device_target_arch((_TensorWithDevice(device),))

    def test_no_recognized_arch_attribute_is_rejected(self):
        with pytest.raises(
            ValueError, match="Unsupported or undetectable TT device architecture"
        ):
            ttl_api._device_target_arch((_TensorWithDevice(object()),))

    def test_no_tensor_args_returns_none(self):
        assert ttl_api._device_target_arch(()) is None

    def test_raising_arch_attribute_is_rejected(self):
        with pytest.raises(
            ValueError, match="Unsupported or undetectable TT device architecture"
        ):
            ttl_api._device_target_arch((_TensorWithDevice(_DeviceWithRaisingArch()),))

    def test_different_device_architectures_are_rejected(self):
        args = (
            _TensorWithDevice(_DeviceWithArchAttribute("Arch.WORMHOLE_B0")),
            _TensorWithDevice(_DeviceWithArchAttribute("Arch.BLACKHOLE")),
        )
        with pytest.raises(ValueError, match="different TT device architectures"):
            ttl_api._device_target_arch(args)


class TestMeshProgramPlacement:
    @pytest.fixture(autouse=True)
    def _patch_tensor_detection(self):
        with mock.patch.object(
            ttl_api, "is_ttnn_tensor", lambda arg: isinstance(arg, _TensorWithDevice)
        ):
            yield

    def test_default_mesh_program_placement_covers_mesh(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((2, 4)))

        placements = ttl_api._default_mesh_program_placements((tensor,))

        assert len(placements) == 1
        assert placements[0].start == (0, 0)
        assert placements[0].end == (1, 3)

    def test_default_mesh_program_placement_skips_single_device(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 1)))

        assert ttl_api._default_mesh_program_placements((tensor,)) is None

    def test_device_domain_mesh_program_placement_covers_domain(self):
        domain = ttl.DeviceDomain((1, 4))

        placements = ttl_api._default_mesh_program_placements_with_domain((), domain)

        assert len(placements) == 1
        assert placements[0].start == (0, 0)
        assert placements[0].end == (0, 3)

    def test_device_domain_mesh_program_placement_matches_mesh_tensor(self):
        domain = ttl.DeviceDomain((1, 4))
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 4)))

        placements = ttl_api._default_mesh_program_placements_with_domain(
            (tensor,), domain
        )

        assert placements == [ttl_api.MeshProgramPlacement((0, 0), (0, 3))]

    def test_device_domain_mesh_program_placement_rejects_mismatch(self):
        domain = ttl.DeviceDomain((1, 2))
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 4)))

        with pytest.raises(ValueError, match="does not match"):
            ttl_api._default_mesh_program_placements_with_domain((tensor,), domain)

    def test_device_domain_mesh_program_placement_supports_product(self):
        domain = ttl.DeviceDomain.product(board=(1,), device=(4,))

        placements = ttl_api._default_mesh_program_placements_with_domain((), domain)

        assert placements == [ttl_api.MeshProgramPlacement((0, 0), (0, 3))]

    def test_explicit_mesh_program_placements_override_full_domain(self):
        domain = ttl.DeviceDomain((4, 8))
        placements = [(0, 0), (3, 7)]

        resolved = ttl_api._resolve_mesh_program_placements((), domain, placements)

        assert resolved == (
            ttl.MeshProgramPlacement((0, 0)),
            ttl.MeshProgramPlacement((3, 7)),
        )

    def test_explicit_mesh_program_placements_are_immutable(self):
        start = [0, 0]
        end = [0, 1]
        coordinate = [1, 1]

        resolved = ttl_api._resolve_mesh_program_placements(
            (),
            ttl.DeviceDomain((2, 2)),
            [ttl.MeshProgramPlacement(start, end), coordinate],
        )
        start[0] = 1
        end[1] = 0
        coordinate[0] = 0

        assert resolved == (
            ttl.MeshProgramPlacement((0, 0), (0, 1)),
            ttl.MeshProgramPlacement((1, 1)),
        )

    def test_explicit_mesh_program_placements_allow_sparse_tensor_domain(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 2)))

        resolved = ttl_api._resolve_mesh_program_placements(
            (tensor,),
            ttl.DeviceDomain((2, 2)),
            [(0, 0), (0, 1)],
        )

        assert resolved == (
            ttl.MeshProgramPlacement((0, 0)),
            ttl.MeshProgramPlacement((0, 1)),
        )

    def test_explicit_mesh_program_placements_stay_inside_tensor_mesh(self):
        tensor = _TensorWithDevice(_DeviceWithMeshShape((1, 2)))

        with pytest.raises(ValueError, match="inside the mesh tensor"):
            ttl_api._resolve_mesh_program_placements(
                (tensor,),
                ttl.DeviceDomain((2, 2)),
                [(1, 0)],
            )

    def test_explicit_mesh_program_placements_cover_pipenet_endpoints(self):
        domain = ttl.DeviceDomain((2, 2))
        required_devices = (
            domain.device_ref((0, 0)),
            domain.device_ref((1, 1)),
        )

        resolved = ttl_api._resolve_mesh_program_placements(
            (),
            domain,
            [ttl.MeshProgramPlacement((0, 0), (1, 1))],
            required_devices=required_devices,
        )

        assert resolved == (ttl.MeshProgramPlacement((0, 0), (1, 1)),)

    def test_explicit_mesh_program_placements_reject_missing_pipenet_endpoint(self):
        domain = ttl.DeviceDomain((2, 2))

        with pytest.raises(ValueError, match=r"missing \[\(1, 1\)\]"):
            ttl_api._resolve_mesh_program_placements(
                (),
                domain,
                [(0, 0)],
                required_devices=(domain.device_ref((1, 1)),),
            )

    def test_product_domain_pipenet_endpoints_use_runtime_coordinate_order(self):
        domain = ttl.DeviceDomain.product(board=(2,), device=(2,))

        resolved = ttl_api._resolve_mesh_program_placements(
            (),
            domain,
            [ttl.MeshProgramPlacement((1, 0), (1, 1))],
            required_devices=(ttl.DeviceRef(board=1, device=0),),
        )

        assert resolved == (ttl.MeshProgramPlacement((1, 0), (1, 1)),)

    @pytest.mark.parametrize("placements", [(), []])
    def test_explicit_mesh_program_placements_reject_empty(self, placements):
        with pytest.raises(ValueError, match="must not be empty"):
            ttl_api._resolve_mesh_program_placements(
                (), ttl.DeviceDomain((1, 2)), placements
            )

    @pytest.mark.parametrize(
        ("placements", "error_type", "message"),
        [
            ("0,0", TypeError, "must be a tuple or list"),
            ([object()], TypeError, "must be coordinate tuples"),
            ([()], ValueError, "start must not be empty"),
            ([(0, "1")], TypeError, "coordinates must be integers"),
            ([(True, 0)], TypeError, "coordinates must be integers"),
        ],
    )
    def test_explicit_mesh_program_placements_reject_invalid_values(
        self, placements, error_type, message
    ):
        with pytest.raises(error_type, match=message):
            ttl_api._resolve_mesh_program_placements(
                (), ttl.DeviceDomain((1, 2)), placements
            )

    @pytest.mark.parametrize(
        "placements",
        [
            [(0, 0), (0, 0)],
            [
                ttl.MeshProgramPlacement((0, 0), (1, 2)),
                ttl.MeshProgramPlacement((1, 1), (2, 3)),
            ],
        ],
        ids=("duplicate-coordinate", "partial-multidimensional-intersection"),
    )
    def test_mesh_program_placements_reject_intersections(self, placements):
        with pytest.raises(ValueError, match="indices 0 and 1 must not overlap"):
            ttl_api.normalize_mesh_program_placements(placements)

    def test_mesh_program_placements_accept_disjoint_inclusive_ranges(self):
        placements = [
            ttl.MeshProgramPlacement((0, 0), (0, 1)),
            ttl.MeshProgramPlacement((0, 2), (0, 3)),
        ]

        assert ttl_api.normalize_mesh_program_placements(placements) == tuple(
            placements
        )

    def test_mesh_program_placements_reject_mixed_ranks_without_extent(self):
        with pytest.raises(ValueError, match="placement 1 has rank 3, expected rank 2"):
            ttl_api.normalize_mesh_program_placements([(0, 0), (0, 0, 0)])

    @pytest.mark.parametrize(
        ("start", "end", "error_type", "message"),
        [
            (0, None, TypeError, "start must be a coordinate tuple"),
            ((-1, 0), None, ValueError, "coordinates must be non-negative"),
            ((0,), (0, 1), ValueError, "same rank"),
            ((0, 1), (0, 0), ValueError, "must not exceed"),
        ],
    )
    def test_mesh_program_placement_rejects_invalid_range(
        self, start, end, error_type, message
    ):
        with pytest.raises(error_type, match=message):
            ttl.MeshProgramPlacement(start, end)

    def test_compile_kernel_forwards_device_domain_to_lowering(self, monkeypatch):
        domain = ttl.DeviceDomain((1, 2))
        calls = []

        def compute_thread():
            pass

        compute_thread._logical_kernel = ttl.KernelKind.COMPUTE
        monkeypatch.setattr(
            ttl_api, "_get_registered_threads", lambda: [compute_thread]
        )
        monkeypatch.setattr(
            ttl_api,
            "_build_operation_pipenets",
            lambda *_: ttl_api._build_pipenet_graph([]),
        )

        def fake_lower_program_to_kernel(**kwargs):
            calls.append(kwargs)
            return "compiled"

        monkeypatch.setattr(
            ttl_api, "_lower_program_to_kernel", fake_lower_program_to_kernel
        )

        def kernel():
            pass

        result = ttl_api._compile_kernel(
            kernel,
            (),
            {},
            (1, 1),
            [],
            [],
            0,
            "L1",
            True,
            0,
            device_domain=domain,
        )

        assert result == "compiled"
        assert calls[0]["device_domain"] is domain
        assert calls[0]["mesh_program_placements"] == (
            ttl_api.MeshProgramPlacement((0, 0), (0, 1)),
        )
        assert calls[0]["logical_kernels"] == [ttl.KernelKind.COMPUTE]

    def test_compile_kernel_rejects_placement_before_lowering_when_endpoint_missing(
        self, monkeypatch
    ):
        domain = ttl.DeviceDomain((1, 2))
        graph = ttl.TransferGraph.edges(domain, edges=[((0, 0), (0, 1))])
        operation_pipenets = ttl_api._build_pipenet_graph([ttl.PipeNet(graph=graph)])

        def compute_thread():
            pass

        compute_thread._logical_kernel = ttl.KernelKind.COMPUTE
        monkeypatch.setattr(
            ttl_api, "_get_registered_threads", lambda: [compute_thread]
        )
        monkeypatch.setattr(
            ttl_api,
            "_build_operation_pipenets",
            lambda *_: operation_pipenets,
        )
        lowering_calls = []
        monkeypatch.setattr(
            ttl_api,
            "_lower_program_to_kernel",
            lambda **kwargs: lowering_calls.append(kwargs),
        )

        with pytest.raises(ValueError, match="cover every PipeNet endpoint"):
            ttl_api._compile_kernel(
                lambda: None,
                (),
                {},
                (1, 1),
                [],
                [],
                0,
                "L1",
                True,
                0,
                device_domain=domain,
                mesh_program_placements=[(0, 0)],
            )

        assert lowering_calls == []

    def test_operation_forwards_device_domain_to_explicit_compiler(self, monkeypatch):
        domain = ttl.DeviceDomain((1, 2))
        placements = [(0, 0)]
        decorator_options = []

        def fake_pykernel_gen(**kwargs):
            decorator_options.append(kwargs)
            return lambda fn: fn

        monkeypatch.setattr(ttl_atom, "pykernel_gen", fake_pykernel_gen)
        monkeypatch.setattr(ttl_atom, "_has_explicit_kernels", lambda _: True)

        @ttl_atom.operation(
            grid=(1, 1),
            device_domain=domain,
            mesh_program_placements=placements,
        )
        def operation():
            pass

        assert decorator_options[0]["device_domain"] is domain
        assert decorator_options[0]["mesh_program_placements"] is placements

    def test_unified_compiler_forwards_device_domain(self, monkeypatch):
        domain = ttl.DeviceDomain((1, 2))
        calls = []

        def fake_compile_atom(*args, **kwargs):
            calls.append((args, kwargs))
            return "compiled"

        monkeypatch.setattr(ttl_atom, "_compile_atom", fake_compile_atom)
        decorator_options = {
            "num_outs": 1,
            "memory_space": "L1",
            "tiled": True,
            "fp32_dest_acc_en": None,
            "dst_full_sync_en": None,
            "math_fidelity": None,
            "device_domain": domain,
            "mesh_program_placements": [(0, 1)],
            "runtime_resource_factory": None,
        }

        result = ttl_atom._compile_unified_operation(
            object(),
            decorator_options,
            (),
            {},
            (1, 1),
            0,
            None,
            ttl.CompilerOptions(),
            0,
        )

        assert result == "compiled"
        assert calls[0][1]["device_domain"] is domain
        assert calls[0][1]["mesh_program_placements"] == [(0, 1)]

    def test_compiled_kernel_forwards_mesh_program_placements(self, monkeypatch):
        placement = ttl.MeshProgramPlacement((0, 0), (0, 3))
        calls = []

        def fake_run_kernel_on_device(**kwargs):
            calls.append(kwargs)
            return "result"

        monkeypatch.setattr(ttl_api, "run_kernel_on_device", fake_run_kernel_on_device)
        compiled_kernel = ttl_api.CompiledTTNNKernel(
            kernel_paths=[],
            kernel_configs=[],
            kernel_arg_specs=[],
            num_tensors=1,
            core_ranges=_CoreRanges(),
            kernel_tensor_indices=[],
            mesh_program_placements=[placement],
        )

        result = compiled_kernel(_TensorWithDevice(_DeviceWithMeshShape((1, 4))))

        assert result == "result"
        assert calls[0]["mesh_program_placements"] == [placement]
        assert calls[0]["fabric_route_cache"] is compiled_kernel._fabric_route_cache


@pytest.mark.parametrize("logical_selectors", [None, [], [None]])
def test_resource_factory_requires_complete_logical_selectors(logical_selectors):
    with pytest.raises(
        ValueError,
        match="runtime_resource_factory requires .*logical-kernel selector",
    ):
        ttl_api.CompiledTTNNKernel(
            kernel_paths=[("kernel.cpp", "compute")],
            kernel_configs=[object()],
            kernel_arg_specs=[[]],
            num_tensors=1,
            core_ranges=object(),
            kernel_tensor_indices=[[]],
            kernel_logical_selectors=logical_selectors,
            operation_name="resource_operation",
            runtime_resource_factory=lambda **_kwargs: ProgramRuntimeResources(),
        )


class TestKernelI32ArrayAttr:
    def test_optional_attribute_may_be_absent(self):
        context = Context()
        ttl_dialect.ensure_dialects_registered(context)

        with context:
            module = Module.parse("module { func.func @reader() { return } }")
            assert (
                ttl_api._get_kernel_optional_i32_array_attr(
                    module, "reader", "ttl.pipe_computed_address_dfb_indices"
                )
                is None
            )

    def test_optional_attribute_empty_array_is_not_missing(self):
        context = Context()
        ttl_dialect.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @reader() attributes {
                    ttl.used_dfb_indices = array<i32>
                  } {
                    return
                  }
                }
                """
            )
            assert (
                ttl_api._get_kernel_optional_i32_array_attr(
                    module, "reader", "ttl.used_dfb_indices"
                )
                == []
            )

    def test_optional_attribute_is_read_when_present(self):
        context = Context()
        ttl_dialect.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @reader() attributes {
                    ttl.pipe_computed_address_dfb_indices = array<i32: 2, 5>
                  } {
                    return
                  }
                }
                """
            )
            assert ttl_api._get_kernel_optional_i32_array_attr(
                module, "reader", "ttl.pipe_computed_address_dfb_indices"
            ) == [2, 5]

    def test_optional_attribute_is_validated_when_present(self):
        context = Context()
        ttl_dialect.ensure_dialects_registered(context)

        with context:
            module = Module.parse(
                """
                module {
                  func.func @reader() attributes {
                    ttl.pipe_computed_address_dfb_indices = 2 : i32
                  } {
                    return
                  }
                }
                """
            )
            with pytest.raises(ValueError, match="Expected DenseI32ArrayAttr"):
                ttl_api._get_kernel_optional_i32_array_attr(
                    module, "reader", "ttl.pipe_computed_address_dfb_indices"
                )

    def test_required_attribute_must_be_present(self):
        context = Context()
        ttl_dialect.ensure_dialects_registered(context)

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
                """
            )
            with pytest.raises(
                ValueError,
                match="Required compiler-generated attribute "
                "'ttl.unpack_to_dest_fp32' is missing",
            ):
                ttl_api._get_kernel_i32_array_attr(
                    module, "compute_kernel", "ttl.unpack_to_dest_fp32"
                )


class TestMathFidelity:
    @pytest.mark.parametrize("math_fidelity", SUPPORTED_MATH_FIDELITIES)
    def test_maps_supported_value(self, math_fidelity):
        fidelity_value = object()
        ttnn_module = mock.Mock()
        setattr(ttnn_module.MathFidelity, math_fidelity, fidelity_value)
        config = mock.Mock()

        ttl_api._set_math_fidelity(config, ttnn_module, math_fidelity)

        assert config.math_fidelity is fidelity_value

    def test_missing_ttnn_value_is_rejected(self):
        ttnn_module = mock.Mock()
        del ttnn_module.MathFidelity.HiFi4

        with pytest.raises(RuntimeError, match="does not provide MathFidelity.HiFi4"):
            ttl_api._set_math_fidelity(mock.Mock(), ttnn_module, "HiFi4")
