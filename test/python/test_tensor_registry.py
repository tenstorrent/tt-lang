# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttl._src.tensor_registry import (
    get_tensor_global_index,
    get_tensor_global_name,
    register_tensor_arguments,
)


def test_register_tensor_arguments_canonicalizes_aliases_to_first_position():
    primary_tensor = object()
    other_tensor = object()

    register_tensor_arguments(
        (
            (primary_tensor, "primary_tensor", 3),
            (other_tensor, "other_tensor", 4),
            (primary_tensor, "alias_tensor", 7),
        )
    )

    assert get_tensor_global_name(primary_tensor) == "primary_tensor"
    assert get_tensor_global_index(primary_tensor) == 3
    assert get_tensor_global_name(other_tensor) == "other_tensor"
    assert get_tensor_global_index(other_tensor) == 4
