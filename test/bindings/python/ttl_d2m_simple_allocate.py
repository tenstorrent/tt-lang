# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# RUN: %python %s | FileCheck %s
#
# Sanity check that the d2m-simple-allocate pass is registered and runnable
# from Python. We run it over a trivial function and check the printed IR for
# the injected ttcore.system_desc module attribute and default device.

from ttmlir import ir
from ttmlir import passmanager
from ttlang.dialects import ttl


def main():
    with ir.Context() as ctx, ir.Location.unknown():
        ttl.ensure_dialects_registered(ctx)

        pm = passmanager.PassManager.parse(
            "builtin.module(ttcore-register-device,func.func(d2m-simple-allocate))"
        )

        module = ir.Module.parse(
            r"""
module {
  func.func @empty() {
    return
  }
}
"""
        )

        pm.run(module.operation)
        print(module)


# CHECK: module attributes {ttcore.system_desc =
# CHECK: ttcore.device @default_device


if __name__ == "__main__":
    main()
