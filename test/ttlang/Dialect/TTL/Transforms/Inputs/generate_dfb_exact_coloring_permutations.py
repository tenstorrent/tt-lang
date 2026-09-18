# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations


DFB_TYPE = "!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>"
PROTOCOL_TYPE = "<[1, 1], !ttcore.tile<32x32, bf16>, 2>"
TENSOR_TYPE = "tensor<1x1x!ttcore.tile<32x32, bf16>>"


def emit_bind(name, physical_index, logical_id):
    print(
        f"    %{name} = ttl.bind_cb {{cb_index = {physical_index}, "
        f"block_count = 2}} {{dfb_id = {logical_id} : index}} : {DFB_TYPE}"
    )


def emit_protocol(name, effect):
    result_name = "reserved" if effect == "reserve" else "waited"
    print(
        f"    %{result_name}_{name} = ttl.cb_{effect} %{name} : "
        f"{PROTOCOL_TYPE} -> {TENSOR_TYPE}"
    )
    completion_effect = "push" if effect == "reserve" else "pop"
    print(f"    ttl.cb_{completion_effect} %{name} : {PROTOCOL_TYPE}")


def emit_module(logical_ids):
    print("module {")
    print("  func.func @exact_coloring_permutation()")
    print("      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,")
    print("                  ttl.base_cta_index = 34 : i32, ttl.crta_indices = []} {")
    for persistent_index in range(30):
        emit_bind(f"persistent{persistent_index}", persistent_index, persistent_index)

    path_names = ("path_a", "path_d", "path_b", "path_c")
    for declaration_index, (name, logical_id) in enumerate(
        zip(path_names, logical_ids), start=30
    ):
        emit_bind(name, declaration_index, logical_id)

    emit_protocol("path_a", "reserve")
    emit_protocol("path_b", "reserve")
    emit_protocol("path_a", "wait")
    emit_protocol("path_c", "reserve")
    emit_protocol("path_b", "wait")
    emit_protocol("path_d", "reserve")
    emit_protocol("path_c", "wait")
    emit_protocol("path_d", "wait")
    print("    return")
    print("  }")
    print("}")


for permutation_index, logical_ids in enumerate(permutations(range(30, 34))):
    if permutation_index:
        print("// -----")
    emit_module(logical_ids)
