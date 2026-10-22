#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the MathInit matrix tests for ttkernel-verify-hardware-config.

Every case places one init and one compute operation in a control-flow shape.
The expected verdict comes from an oracle that enumerates execution paths
with up to two loop iterations, independently of the transfer functions used
by the compiler. Cases the oracle accepts go to the FileCheck file; the rest
go to the --verify-diagnostics file.

Usage: gen_verify_hardware_config.py <output-directory>
"""

import itertools
import pathlib
import sys

CB = "!ttkernel.cb<4, !ttcore.tile<32x32, f32>>"
VALID_FILE = "verify_hardware_config_matrix.mlir"
INVALID_FILE = "verify_hardware_config_matrix_invalid.mlir"

_ids = itertools.count()


class Op:
    """A TTKernel operation that reads or writes MathInit.

    `descriptor` is (init name, key operands, key attributes). A write with a
    None descriptor leaves the configuration unknown.
    """

    def __init__(self, text, is_write, descriptor):
        self.id = next(_ids)
        self.text = text
        self.is_write = is_write
        self.descriptor = descriptor


class If:
    def __init__(self, then_body, else_body):
        self.then_body = then_body
        self.else_body = else_body


class For:
    """A loop with two iterations when static, otherwise zero or more."""

    def __init__(self, static, body):
        self.static = static
        self.body = body


class While:
    def __init__(self, before, after):
        self.before = before
        self.after = after


def write(text, descriptor):
    return lambda: Op(text, True, descriptor)


def read(text, descriptor):
    return lambda: Op(text, False, descriptor)


def descriptor(init, operands=(), attributes=()):
    return ("ttkernel." + init, tuple(operands), tuple(sorted(attributes)))


COPY_INIT = descriptor("copy_tile_init", ["a"])
EXP_INIT = descriptor("exp_tile_init", [], [("approx", "true")])
ADD_INIT = descriptor("add_tiles_init", ["a", "b"])
BCAST_INIT = descriptor("unary_bcast_init", ["a"], [("bcast_type", "col")])
REDUCE_INIT = descriptor(
    "reduce_init", ["a", "b"], [("reduce_type", "sum"), ("reduce_dim", "col")]
)

# Each consumer lists writers by their relation to the required init:
# `equal` is the required init, `unkeyed` differs only in operands outside
# the key, and `key` differs in a key operand or attribute.
CONSUMERS = {
    "copy": {
        "read": read(
            f"ttkernel.copy_tile(%cb_a, %c0, %c0) : ({CB}, index, index) -> ()",
            COPY_INIT,
        ),
        "equal": write(f"ttkernel.copy_tile_init(%cb_a) : ({CB}) -> ()", COPY_INIT),
        "key": write(
            f"ttkernel.copy_tile_init(%cb_b) : ({CB}) -> ()",
            descriptor("copy_tile_init", ["b"]),
        ),
    },
    "exp": {
        "read": read(
            "ttkernel.exp_tile(%c0) {approx = true} : (index) -> ()", EXP_INIT
        ),
        "equal": write("ttkernel.exp_tile_init() {approx = true} : () -> ()", EXP_INIT),
        "key": write(
            "ttkernel.exp_tile_init() {approx = false} : () -> ()",
            descriptor("exp_tile_init", [], [("approx", "false")]),
        ),
    },
    "add": {
        "read": read(
            "ttkernel.add_tiles(%cb_a, %cb_b, %c0, %c0, %c0) : "
            f"({CB}, {CB}, index, index, index) -> ()",
            ADD_INIT,
        ),
        "equal": write(
            f"ttkernel.add_tiles_init(%cb_a, %cb_b) : ({CB}, {CB}) -> ()", ADD_INIT
        ),
        "key": write(
            f"ttkernel.add_tiles_init(%cb_b, %cb_a) : ({CB}, {CB}) -> ()",
            descriptor("add_tiles_init", ["b", "a"]),
        ),
    },
    "bcast": {
        "read": read(
            "ttkernel.unary_bcast(%cb_a, %c0, %c0, <col>) : "
            f"({CB}, index, index) -> ()",
            BCAST_INIT,
        ),
        "equal": write(
            f"ttkernel.unary_bcast_init(%cb_a, %cb_c, <col>) : ({CB}, {CB}) -> ()",
            BCAST_INIT,
        ),
        "unkeyed": write(
            f"ttkernel.unary_bcast_init(%cb_a, %cb_b, <col>) : ({CB}, {CB}) -> ()",
            BCAST_INIT,
        ),
        "key": write(
            f"ttkernel.unary_bcast_init(%cb_a, %cb_c, <row>) : ({CB}, {CB}) -> ()",
            descriptor("unary_bcast_init", ["a"], [("bcast_type", "row")]),
        ),
    },
    "reduce": {
        "read": read(
            "ttkernel.reduce_tile(%cb_a, %cb_b, %c0, %c0, %c0, <reduce_sum>, "
            f"<reduce_dim_col>) : ({CB}, {CB}, index, index, index) -> ()",
            REDUCE_INIT,
        ),
        "equal": write(
            "ttkernel.reduce_init(%cb_a, %cb_b, %cb_c, <reduce_sum>, "
            f"<reduce_dim_col>) : ({CB}, {CB}, {CB}) -> ()",
            REDUCE_INIT,
        ),
        "unkeyed": write(
            "ttkernel.reduce_init(%cb_a, %cb_b, %cb_a, <reduce_sum>, "
            f"<reduce_dim_col>) : ({CB}, {CB}, {CB}) -> ()",
            REDUCE_INIT,
        ),
        "key": write(
            "ttkernel.reduce_init(%cb_a, %cb_b, %cb_c, <reduce_sum>, "
            f"<reduce_dim_row>) : ({CB}, {CB}, {CB}) -> ()",
            descriptor(
                "reduce_init",
                ["a", "b"],
                [("reduce_type", "sum"), ("reduce_dim", "row")],
            ),
        ),
    },
}

OTHER_INIT = write("ttkernel.fill_tile_init() : () -> ()", descriptor("fill_tile_init"))
FULL_INIT = write(f"ttkernel.init_sfpu(%cb_a, %cb_c) : ({CB}, {CB}) -> ()", None)

# Each placement receives factories for the reader `r`, the writer under test
# `w`, and the required init `e`, and returns the function body.
PLACEMENTS = {
    "straight": lambda r, w, e: [w(), r()],
    "reset_after_write": lambda r, w, e: [w(), FULL_INIT(), r()],
    "if_both": lambda r, w, e: [If([w()], [w()]), r()],
    "if_then": lambda r, w, e: [If([w()], []), r()],
    "if_join_with_required": lambda r, w, e: [e(), If([w()], []), r()],
    "for_hoisted_static": lambda r, w, e: [w(), For(True, [r()])],
    "for_hoisted_dynamic": lambda r, w, e: [w(), For(False, [r()])],
    "for_inside": lambda r, w, e: [For(False, [w(), r()])],
    "for_backedge": lambda r, w, e: [e(), For(False, [r(), w()])],
    "for_backedge_without_entry": lambda r, w, e: [For(True, [r(), w()])],
    "for_exit_static": lambda r, w, e: [For(True, [w()]), r()],
    "for_exit_dynamic": lambda r, w, e: [e(), For(False, [w()]), r()],
    "while_after_writes": lambda r, w, e: [e(), While([], [w()]), r()],
    "while_before_writes": lambda r, w, e: [While([w()], []), r()],
    "while_before_reads": lambda r, w, e: [w(), While([r()], [])],
    "while_after_reads": lambda r, w, e: [While([], [w(), r()])],
}

# Placements whose verdict does not depend on the writer relation.
RELATION_INDEPENDENT = {"missing": lambda r, w, e: [r()]}


def simulate(nodes, states, incoming):
    """Return the exit states of `nodes`; record entry states of reads."""
    for node in nodes:
        states = step(node, states, incoming)
    return states


def step(node, states, incoming):
    if isinstance(node, Op):
        if node.is_write:
            return {(node.descriptor, node.id)}
        incoming.setdefault(node.id, set()).update(states)
        return states
    if isinstance(node, If):
        return simulate(node.then_body, states, incoming) | simulate(
            node.else_body, states, incoming
        )
    if isinstance(node, For):
        first = simulate(node.body, states, incoming)
        second = simulate(node.body, first, incoming)
        return second if node.static else states | first | second
    if isinstance(node, While):
        exits = set()
        current = states
        for _ in range(3):
            before = simulate(node.before, current, incoming)
            exits |= before
            current = simulate(node.after, before, incoming)
        return exits
    raise TypeError(node)


def merge(states):
    """Merge path states the way the compiler reports them.

    The descriptor survives only when every path agrees. Up to two writers,
    in program order, are kept either way so a conflict can still be named.
    """
    descriptors = {state[0] for state in states}
    writers = []
    for writer in sorted(state[1] for state in states if state[1] is not None):
        if writer not in writers:
            writers.append(writer)
    descriptor = descriptors.pop() if len(descriptors) == 1 else None
    return descriptor, writers[:2]


def diagnose(op, states):
    """Return (error, [(writer id, note), ...]) for a failing read, else None."""
    merged_descriptor, writers = merge(states)
    if merged_descriptor == op.descriptor:
        return None
    by_writer = {
        writer: descriptor for descriptor, writer in states if writer is not None
    }
    error = f"requires MATH configuration from '{op.descriptor[0]}'"
    if merged_descriptor is None:
        error += " but it is not established on every incoming path"
    elif merged_descriptor[0] != op.descriptor[0]:
        error += f" but '{merged_descriptor[0]}' is configured"
    else:
        error += " but it is configured with different operands or attributes"
    notes = [
        (
            writer,
            (
                "MATH configuration reset here"
                if by_writer[writer] is None
                else "configured here"
            ),
        )
        for writer in writers
    ]
    return error, notes


def emit(nodes, indent, errors, notes, lines, counter):
    pad = "  " * indent
    for node in nodes:
        if isinstance(node, Op):
            for note in notes.get(node.id, []):
                lines.append(f"{pad}// expected-note @below {{{{{note}}}}}")
            if node.id in errors:
                lines.append(f"{pad}// expected-error @below {{{{{errors[node.id]}}}}}")
            lines.append(pad + node.text)
        elif isinstance(node, If):
            lines.append(f"{pad}scf.if %cond {{")
            emit(node.then_body, indent + 1, errors, notes, lines, counter)
            if node.else_body:
                lines.append(f"{pad}}} else {{")
                emit(node.else_body, indent + 1, errors, notes, lines, counter)
            lines.append(f"{pad}}}")
        elif isinstance(node, For):
            iv = f"%i{next(counter)}"
            upper = "%c2" if node.static else "%n"
            lines.append(f"{pad}scf.for {iv} = %c0 to {upper} step %c1 {{")
            emit(node.body, indent + 1, errors, notes, lines, counter)
            lines.append(f"{pad}}}")
        elif isinstance(node, While):
            lines.append(f"{pad}scf.while : () -> () {{")
            emit(node.before, indent + 1, errors, notes, lines, counter)
            lines.append(f"{pad}  scf.condition(%cond)")
            lines.append(f"{pad}}} do {{")
            emit(node.after, indent + 1, errors, notes, lines, counter)
            lines.append(f"{pad}  scf.yield")
            lines.append(f"{pad}}}")


def render_case(name, description, nodes, valid):
    incoming = {}
    simulate(nodes, {(None, None)}, incoming)
    errors, notes = {}, {}
    for op_id, states in incoming.items():
        op = find_op(nodes, op_id)
        result = diagnose(op, states)
        if result is None:
            continue
        error, writer_notes = result
        errors[op_id] = error
        for writer, note in writer_notes:
            notes.setdefault(writer, []).append(note)
    if (not errors) != valid:
        return None

    lines = [f"// {description}"]
    if valid:
        lines.append(f"// CHECK-LABEL: func.func @{name}")
    lines += [
        f"func.func @{name}(%cond: i1, %n: index) {{",
        f"  %cb_a = ttkernel.get_compile_time_arg_val(0) : () -> {CB}",
        f"  %cb_b = ttkernel.get_compile_time_arg_val(1) : () -> {CB}",
        f"  %cb_c = ttkernel.get_compile_time_arg_val(2) : () -> {CB}",
        "  %c0 = arith.constant 0 : index",
        "  %c1 = arith.constant 1 : index",
        "  %c2 = arith.constant 2 : index",
    ]
    emit(nodes, 1, errors, notes, lines, itertools.count())
    lines += ["  func.return", "}"]
    return "\n".join(lines)


def find_op(nodes, op_id):
    for node in nodes:
        if isinstance(node, Op):
            if node.id == op_id:
                return node
            continue
        children = {
            If: lambda n: n.then_body + n.else_body,
            For: lambda n: n.body,
            While: lambda n: n.before + n.after,
        }[type(node)](node)
        found = find_op(children, op_id)
        if found is not None:
            return found
    return None


def cases():
    for consumer_name, consumer in CONSUMERS.items():
        writers = {
            relation: consumer[relation]
            for relation in ("equal", "unkeyed", "key")
            if relation in consumer
        }
        writers["other_init"] = OTHER_INIT
        writers["full_init"] = FULL_INIT
        for placement, build in RELATION_INDEPENDENT.items():
            yield (
                f"{consumer_name}_{placement}",
                f"Consumer {consumer_name}, no init.",
                build(consumer["read"], None, consumer["equal"]),
            )
        for (relation, writer), (placement, build) in itertools.product(
            writers.items(), PLACEMENTS.items()
        ):
            yield (
                f"{consumer_name}_{relation}_{placement}",
                f"Consumer {consumer_name}, {relation} writer, {placement}.",
                build(consumer["read"], writer, consumer["equal"]),
            )


def render_file(valid):
    if valid:
        header = [
            "// RUN: ttlang-opt %s --split-input-file "
            "--ttkernel-verify-hardware-config | FileCheck %s",
            "// Generated by Inputs/gen_verify_hardware_config.py; do not edit.",
            "// Summary: MathInit matrix cases that the path-enumerating oracle "
            "accepts.",
        ]
    else:
        header = [
            "// RUN: ttlang-opt %s --split-input-file "
            "--ttkernel-verify-hardware-config --verify-diagnostics",
            "// Generated by Inputs/gen_verify_hardware_config.py; do not edit.",
            "// Summary: MathInit matrix cases that the path-enumerating oracle "
            "rejects.",
        ]
    bodies = [
        text
        for text in (
            render_case(name, description, nodes, valid)
            for name, description, nodes in cases()
        )
        if text is not None
    ]
    header.append(f"// Cases: {len(bodies)}.")
    return "\n".join(header) + "\n\n" + "\n\n// -----\n\n".join(bodies) + "\n", len(
        bodies
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    output = pathlib.Path(sys.argv[1])
    total = 0
    for valid, name in ((True, VALID_FILE), (False, INVALID_FILE)):
        text, count = render_file(valid)
        (output / name).write_text(text)
        total += count
    expected = sum(
        len(RELATION_INDEPENDENT)
        + len(PLACEMENTS) * (len({"equal", "unkeyed", "key"} & consumer.keys()) + 2)
        for consumer in CONSUMERS.values()
    )
    if total != expected:
        sys.exit(f"generated {total} cases, expected {expected}")


if __name__ == "__main__":
    main()
