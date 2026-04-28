# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test that simulator modules have no mutable module-level globals.

This test enforces the zero-globals architecture where all mutable state
is stored in greenlet-local context (context.py) rather than module-level
variables.
"""

import importlib
import inspect
import types
from pathlib import Path
from typing import Any


def is_acceptable_module_attribute(name: str, obj: Any) -> tuple[bool, str]:
    """
    Check if a module-level attribute is acceptable (not a mutable global).

    Returns:
        (is_acceptable, reason) - True if acceptable with reason, False otherwise
    """
    # Python metadata dicts are acceptable
    if name in ("__annotations__", "__builtins__", "__warningregistry__"):
        return True, "Python metadata"

    # Python module metadata strings (provided by runtime)
    if name in (
        "__name__",
        "__doc__",
        "__file__",
        "__package__",
        "__cached__",
        "__loader__",
        "__spec__",
    ):
        return True, "Python module metadata"

    # Python package metadata (for packages, not modules)
    if name in ("__path__", "__all__"):
        return True, "Python package metadata"

    # Constants (uppercase names, including _UPPERCASE) are acceptable by convention
    if name.isupper() or (
        name.startswith("_") and len(name) > 1 and name[1:].isupper()
    ):
        return True, "uppercase constant name"

    # Functions are acceptable (including private functions)
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return True, "function/method"

    # Classes (including dataclasses, enums, protocols) are acceptable
    if inspect.isclass(obj):
        return True, "class/type"

    # Modules (imports) are acceptable
    if inspect.ismodule(obj):
        return True, "imported module"

    # Built-in types are acceptable
    if isinstance(obj, type):
        return True, "type object"

    # Type aliases and typing constructs
    if any(
        keyword in str(type(obj).__name__)
        for keyword in ["typing", "Generic", "UnionType"]
    ):
        return True, "type annotation"

    # Generic types like List[str], Dict[str, int]
    if hasattr(obj, "__origin__"):
        return True, "generic type annotation"

    # TypeVar (used for generic type parameters)
    from typing import TypeVar as TypeVarType

    if isinstance(obj, type(TypeVarType("T"))):
        return True, "TypeVar for generics"

    # Everything else is NOT acceptable - default to rejecting
    # This includes: mutable dicts, lists, sets, custom objects, etc.
    if isinstance(obj, (dict, list, set)):
        return False, f"mutable {type(obj).__name__}"

    return False, f"unrecognized type ({type(obj).__name__})"


def test_no_mutable_module_globals():
    """Test that simulator modules have no mutable module-level globals."""

    # Dynamically discover all Python modules in python/sim/
    sim_dir = Path("python/sim")
    simulator_modules = []

    # Add the package itself (maps to python/sim/__init__.py)
    simulator_modules.append("python.sim")

    # Add all other .py files as submodules
    # (Skip __init__.py since we already added the package above)
    for py_file in sim_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            module_name = f"python.sim.{py_file.stem}"
            simulator_modules.append(module_name)

    # Sort for consistent ordering
    simulator_modules.sort()

    violations = []

    # Attribute exceptions: specific module.attribute combinations allowed
    # These should be kept minimal and documented with reasons
    attribute_exceptions = {
        (
            "python.sim",
            "ttl",
        ),  # Custom namespace object (_TTLNamespace) for TTL API wrapper
        (
            "python.sim.ttnnsim",
            "bfloat8_b",
        ),  # Immutable singleton dtype sentinel (no native torch equivalent)
    }

    for module_name in simulator_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            # Skip modules that can't be imported (might have dependencies)
            continue

        # Inspect all module-level attributes
        for name in dir(module):
            # Check if this specific attribute is excepted
            if (module_name, name) in attribute_exceptions:
                continue

            # Skip attributes that come from imports (check if defined in this module)
            try:
                obj = getattr(module, name)
            except AttributeError:
                continue

            # Check if this attribute was defined in this module
            # (not inherited from imports)
            if hasattr(obj, "__module__"):
                obj_module = getattr(obj, "__module__", None)
                # Skip if object comes from a different module (it's an import)
                if obj_module and obj_module != module_name:
                    continue

            # Check if acceptable
            acceptable, reason = is_acceptable_module_attribute(name, obj)

            if not acceptable:
                violations.append(
                    {
                        "module": module_name,
                        "name": name,
                        "type": type(obj).__name__,
                        "reason": reason,
                        "value_preview": str(obj)[:100],  # First 100 chars
                    }
                )

    # Report violations
    if violations:
        error_msg = ["Found mutable module-level globals:\n"]
        for v in violations:
            error_msg.append(
                f"  {v['module']}.{v['name']}: {v['reason']}\n"
                f"    Type: {v['type']}\n"
                f"    Value: {v['value_preview']}\n"
            )
        raise AssertionError("".join(error_msg))
