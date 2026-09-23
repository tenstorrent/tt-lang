# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixed, movable, alias, domain, and launch SRAM requirements."""

from dataclasses import replace

import pytest

from ttl._sram_requirements import (
    PersistentSRAMDeclaration,
    PreparedSRAMOperation,
    SRAMArenaBinding,
    SRAMAddressing,
    SRAMEqualBaseGroup,
    SRAMLifetime,
    SRAMLocation,
    SRAMLocationRequirement,
    SRAMOwner,
    SRAMOwnerKind,
    SRAMOwnership,
    SRAMStorageRequirement,
    SRAMUse,
    SRAMUseKind,
    prepare_persistent_storage,
    prepare_sram_operation,
)
from ttl.dataflow_buffer import DFBStorageSegment, PhysicalDFBConfig, SRAMCoreLayout


class CoreCoord:
    def __init__(self, column, row):
        self.x = column
        self.y = row


class MeshCoordinate:
    def __init__(self, coordinate):
        self.coordinate = tuple(coordinate)

    def __iter__(self):
        return iter(self.coordinate)


class TTNN:
    CoreCoord = CoreCoord
    MeshCoordinate = MeshCoordinate

    @staticmethod
    def get_l1_alignment():
        return 16

    @staticmethod
    def corerange_to_cores(grid, row_wise=False):
        del row_wise
        return tuple(CoreCoord(*core) for core in grid)


class Tile:
    tile_shape = (32, 32)

    @staticmethod
    def get_tile_size(dtype):
        return 2048 if dtype == "bfloat16" else 4096


class Tensor:
    def __init__(
        self,
        base,
        *,
        per_core=False,
        devices=((0, 0),),
        cores=((0, 0), (1, 0)),
        shard_shape=(32, 64),
    ):
        self.base = base
        self.per_core = per_core
        self.devices = devices
        self.cores = cores
        self.shard_shape = shard_shape
        self.dtype = "bfloat16"
        self.layout = "TILE"

    def buffer_address(self):
        return self.base

    def is_per_core_allocated(self):
        return self.per_core

    def device_coords(self):
        return self.devices

    def get_tile(self):
        return Tile()

    def memory_config(self):
        shard_spec = type(
            "ShardSpec", (), {"grid": self.cores, "shape": self.shard_shape}
        )()
        return type(
            "MemoryConfig",
            (),
            {
                "buffer_type": "L1",
                "memory_layout": "HEIGHT_SHARDED",
                "shard_spec": shard_spec,
            },
        )()

    def experimental_per_core_buffer_address(self, device, core):
        device_offset = sum(device) * 0x10000
        core_offset = (core.y * 8 + core.x) * 0x1000
        return self.base + device_offset + core_offset


def compiler_config(dfb_index, control_offset, payload_offset, payload_bytes):
    return PhysicalDFBConfig(
        dfb_index=dfb_index,
        num_tiles=1,
        data_format="bfloat16",
        block_count=1,
        page_size=2048,
        tile=(32, 32),
        storage_index=dfb_index,
        l1_offset=control_offset,
        l1_payload_offset=payload_offset,
        l1_allocation_bytes=payload_bytes,
    )


def tensor_config(dfb_index, tensor_index, nodes, byte_offset=0, byte_size=2048):
    return PhysicalDFBConfig(
        dfb_index=dfb_index,
        num_tiles=1,
        data_format="bfloat16",
        block_count=1,
        page_size=2048,
        tile=(32, 32),
        storage_segments=(
            DFBStorageSegment(
                nodes=nodes,
                tensor_index=tensor_index,
                byte_offset=byte_offset,
                byte_size=byte_size,
            ),
        ),
    )


def test_uniform_arena_records_each_dfb_use_once():
    operation = prepare_sram_operation(
        name="uniform",
        tensors=(),
        configs=(
            compiler_config(0, 0, 64, 2048),
            compiler_config(1, 8, 2112, 4096),
        ),
        cores=((1, 0), (0, 0)),
        ttnn_api=TTNN,
    )

    assert len(operation.requirements) == 1
    requirement = operation.requirements[0]
    assert requirement.owner == SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, 0)
    assert requirement.max_extent_bytes == 6208
    assert requirement.ownership is SRAMOwnership.MOVABLE
    assert requirement.lifetime is SRAMLifetime.INVOCATION
    assert operation.arena_bytes_by_core() == {(0, 0): 6208, (1, 0): 6208}
    assert [use.kind for use in operation.uses] == [
        SRAMUseKind.DFB_CONTROL,
        SRAMUseKind.DFB_PAYLOAD,
        SRAMUseKind.DFB_CONTROL,
        SRAMUseKind.DFB_PAYLOAD,
    ]
    assert [use.segment_index for use in operation.uses] == [None, 0, None, 0]


def test_fixed_tensor_and_dfb_aliases_share_one_requirement():
    tensor = Tensor(0x4000)
    operation = prepare_sram_operation(
        name="aliases",
        tensors=(tensor,),
        configs=(
            tensor_config(0, 0, ((0, 0), (1, 0)), byte_size=4096),
            tensor_config(1, 0, ((0, 0),), byte_offset=2048),
        ),
        cores=((0, 0), (1, 0)),
        ttnn_api=TTNN,
    )

    assert len(operation.requirements) == 1
    requirement = operation.requirements[0]
    assert requirement.owner == SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, 0)
    assert requirement.max_extent_bytes == 4096
    assert {value.fixed_base for value in requirement.location_requirements} == {0x4000}
    assert requirement.ownership is SRAMOwnership.FIXED
    assert len(operation.uses) == 2
    assert {use.requirement_index for use in operation.uses} == {0}


def test_fixed_tensor_requirement_covers_unreferenced_storage():
    tensor = Tensor(
        0x4000,
        cores=((0, 0), (1, 0), (2, 0)),
        shard_shape=(32, 96),
    )
    operation = prepare_sram_operation(
        name="fixed_extent",
        tensors=(tensor,),
        configs=(tensor_config(0, 0, ((0, 0),)),),
        cores=((0, 0),),
        ttnn_api=TTNN,
    )

    requirement = operation.requirements[0]
    assert requirement.max_extent_bytes == 6144
    assert len(requirement.equal_base_groups[0].locations) == 3
    assert operation.uses[0].byte_size == 2048


def test_compiler_arena_and_fixed_tensor_have_distinct_owners():
    config = compiler_config(0, 0, 64, 2048)
    config = replace(
        config,
        storage_segments=(
            DFBStorageSegment(
                nodes=((0, 0),),
                tensor_index=0,
                byte_offset=0,
                byte_size=2048,
            ),
        ),
        l1_payload_offset=None,
        l1_allocation_bytes=None,
    )
    operation = prepare_sram_operation(
        name="mixed",
        tensors=(Tensor(0x8000, cores=((0, 0),), shard_shape=(32, 32)),),
        configs=(config,),
        cores=((0, 0),),
        ttnn_api=TTNN,
    )

    assert [requirement.owner.kind for requirement in operation.requirements] == [
        SRAMOwnerKind.COMPILER_ARENA,
        SRAMOwnerKind.TENSOR_ARGUMENT,
    ]
    assert operation.requirements[0].max_extent_bytes == 8
    assert operation.requirements[1].max_extent_bytes == 2048
    assert not operation.requirements[0].location_requirements[0].payload_present


def test_per_core_tensor_has_one_fixed_interval_per_location():
    tensor = Tensor(0x1000, per_core=True, devices=((0, 0), (1, 0)))
    operation = prepare_sram_operation(
        name="per_core",
        tensors=(tensor,),
        configs=(tensor_config(0, 0, ((0, 0), (1, 0))),),
        cores=((0, 0), (1, 0)),
        ttnn_api=TTNN,
    )

    requirement = operation.requirements[0]
    assert requirement.equal_base_groups == ()
    assert tuple(value.fixed_base for value in requirement.location_requirements) == (
        0x1000,
        0x2000,
        0x11000,
        0x12000,
    )


def test_per_core_compiler_layout_preserves_independent_arena_extents():
    config = compiler_config(0, 0, 64, 2048)
    config = replace(
        config,
        sram_core_layouts=(
            SRAMCoreLayout((0, 0), 64, True, 2112, 0),
            SRAMCoreLayout((1, 0), 0, False, 64, 1),
        ),
    )
    operation = prepare_sram_operation(
        name="domains",
        tensors=(),
        configs=(config,),
        cores=((0, 0), (1, 0)),
        ttnn_api=TTNN,
    )

    assert operation.arena_bytes_by_core() == {(0, 0): 2112, (1, 0): 64}
    assert [requirement.max_extent_bytes for requirement in operation.requirements] == [
        2112,
        64,
    ]
    assert [use.kind for use in operation.uses] == [
        SRAMUseKind.DFB_CONTROL,
        SRAMUseKind.DFB_PAYLOAD,
        SRAMUseKind.DFB_CONTROL,
    ]


def test_equal_base_group_preserves_location_extents_and_payload_presence():
    config = replace(
        compiler_config(0, 0, 64, 2048),
        sram_core_layouts=(
            SRAMCoreLayout((0, 0), 64, True, 2112, 0),
            SRAMCoreLayout((1, 0), 0, False, 64, 0),
        ),
    )

    operation = prepare_sram_operation(
        name="variable_capacity",
        tensors=(),
        configs=(config,),
        cores=((0, 0), (1, 0)),
        ttnn_api=TTNN,
    )

    requirement = operation.requirements[0]
    assert [
        (value.extent_bytes, value.payload_present)
        for value in requirement.location_requirements
    ] == [(2112, True), (64, False)]
    assert requirement.equal_base_groups == (SRAMEqualBaseGroup(requirement.locations),)
    assert [use.locations for use in operation.uses] == [
        requirement.locations,
        (requirement.locations[0],),
    ]


@pytest.mark.parametrize(
    ("addressing", "equal_group_count"),
    [(SRAMAddressing.UNIFORM, 1), (SRAMAddressing.PER_CORE, 0)],
)
def test_persistent_declaration_uses_common_requirement_model(
    addressing, equal_group_count
):
    storage = prepare_persistent_storage(
        (
            PersistentSRAMDeclaration(
                extent_bytes=4096,
                alignment_bytes=16,
                cores=((1, 0), (0, 0)),
                addressing=addressing,
            ),
        )
    )

    requirement = storage.requirements[0]
    assert requirement.owner == SRAMOwner(SRAMOwnerKind.PERSISTENT_DECLARATION, 0)
    assert requirement.max_extent_bytes == 4096
    assert requirement.ownership is SRAMOwnership.MOVABLE
    assert requirement.lifetime is SRAMLifetime.PERSISTENT
    assert len(requirement.equal_base_groups) == equal_group_count


def test_requirement_rejects_overlapping_equal_groups_and_misaligned_base():
    location = SRAMLocation((), (0, 0))
    other_location = SRAMLocation((), (1, 0))
    locations = (
        SRAMLocationRequirement(location, 32, True, 0x1000),
        SRAMLocationRequirement(other_location, 32, True, 0x1000),
    )
    group = SRAMEqualBaseGroup((location, other_location))
    with pytest.raises(ValueError, match="must be disjoint"):
        SRAMStorageRequirement(
            SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, 0),
            locations,
            16,
            (group, group),
            SRAMOwnership.FIXED,
            SRAMLifetime.EXTERNAL,
        )
    with pytest.raises(ValueError, match="does not satisfy"):
        SRAMStorageRequirement(
            SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, 0),
            (SRAMLocationRequirement(location, 32, True, 0x1001),),
            16,
            (),
            SRAMOwnership.FIXED,
            SRAMLifetime.EXTERNAL,
        )


def test_operation_rejects_unbound_and_overlapping_arenas():
    first_location = SRAMLocation((), (0, 0))
    second_location = SRAMLocation((), (1, 0))
    requirements = tuple(
        SRAMStorageRequirement(
            SRAMOwner(SRAMOwnerKind.COMPILER_ARENA, index),
            (SRAMLocationRequirement(location, 64, True),),
            16,
            (),
            SRAMOwnership.MOVABLE,
            SRAMLifetime.INVOCATION,
        )
        for index, location in enumerate((first_location, second_location))
    )
    with pytest.raises(ValueError, match="no arena binding"):
        PreparedSRAMOperation(
            "missing",
            requirements,
            (),
            (SRAMArenaBinding(0, ((0, 0),)),),
        )

    overlapping = tuple(
        replace(
            requirement,
            location_requirements=(SRAMLocationRequirement(first_location, 64, True),),
        )
        for requirement in requirements
    )
    with pytest.raises(ValueError, match="multiple compiler arenas"):
        PreparedSRAMOperation(
            "overlap",
            overlapping,
            (),
            (
                SRAMArenaBinding(0, ((0, 0),)),
                SRAMArenaBinding(1, ((0, 0),)),
            ),
        )


def test_operation_rejects_use_outside_extent_and_domain():
    location = SRAMLocation((), (0, 0))
    other_location = SRAMLocation((), (1, 0))
    requirement = SRAMStorageRequirement(
        SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, 0),
        (SRAMLocationRequirement(location, 64, True, 0x1000),),
        16,
        (),
        SRAMOwnership.FIXED,
        SRAMLifetime.EXTERNAL,
    )
    outside_extent = SRAMUse(
        SRAMUseKind.DFB_PAYLOAD,
        0,
        0,
        0,
        48,
        32,
        (location,),
    )
    with pytest.raises(ValueError, match="exceeds its requirement"):
        PreparedSRAMOperation("extent", (requirement,), (outside_extent,), ())

    outside_domain = replace(outside_extent, byte_offset=0, locations=(other_location,))
    with pytest.raises(ValueError, match="outside its requirement"):
        PreparedSRAMOperation("domain", (requirement,), (outside_domain,), ())


def test_operation_rejects_payload_use_on_control_only_location():
    location = SRAMLocation((), (0, 0))
    requirement = SRAMStorageRequirement(
        SRAMOwner(SRAMOwnerKind.TENSOR_ARGUMENT, 0),
        (SRAMLocationRequirement(location, 64, False, 0x1000),),
        16,
        (),
        SRAMOwnership.FIXED,
        SRAMLifetime.EXTERNAL,
    )
    payload_use = SRAMUse(
        SRAMUseKind.DFB_PAYLOAD,
        0,
        0,
        0,
        0,
        32,
        (location,),
    )

    with pytest.raises(ValueError, match="has no storage"):
        PreparedSRAMOperation("control_only", (requirement,), (payload_use,), ())


def test_per_core_tensor_requires_device_coordinates():
    tensor = Tensor(0x1000, per_core=True, devices=())
    with pytest.raises(ValueError, match="logical device coordinates"):
        prepare_sram_operation(
            name="missing_device",
            tensors=(tensor,),
            configs=(tensor_config(0, 0, ((0, 0),)),),
            cores=((0, 0),),
            ttnn_api=TTNN,
        )
