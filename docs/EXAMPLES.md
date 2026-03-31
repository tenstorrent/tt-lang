# Examples: simulator, pytest, and hardware

This document lists kernel examples under `examples/` and `examples/tutorial/`, how they are exercised, and why some are excluded from the default hardware compile-and-run step.

## How to run

**Simulator (host):** from a checkout with the normal venv, `source build/env/activate`.

| Mode | Command |
| --- | --- |
| Simulator CLI | `./bin/ttlang-sim examples/<name>.py` |
| Simulator pytest | `python -m pytest test/sim/test_examples.py -v` |

**Compiler and device (Docker):** run `python` inside the device container using the Docker toolchain venv (`build-docker/env/activate`), not `TTLANG_COMPILE_ONLY`. That mode skips kernel execution and does not validate an example end-to-end.

Example (adjust container name, workdir, and script path to match the machine):

```bash
sudo docker exec -w /path/to/tt-lang-cursor <container> bash -c \
  'source build-docker/env/activate && python examples/<name>.py'
```

For pytest against hardware-backed tests, the same pattern applies with `python -m pytest ...` instead of a single example script.

The hardware CI batch (`.github/scripts/compile-and-run-examples.sh`) uses the same idea: `python3` on each script with the compiler stack and device available. It only includes files that contain `@ttl.operation` under `examples/*.py`, `examples/tutorial/**/*.py`, and `examples/errors/**/*.py`, and that do **not** opt out with `TTLANG_HARDWARE_CI: skip-compiler` in the first 80 lines.

## Matrix

**Sim (pytest `test_example_cli`)** means the script is listed in `test/sim/test_examples.py` and is expected to exit 0 (except where noted). **Device / CI** means a full `python` run with the compiler stack on hardware (Docker pattern above), unless skipped. **Backend** is `sim` if the file uses `from sim import ttl, ttnn`; otherwise it uses real `import ttl` and host `ttnn` (device-backed for tensor creation where applicable).

| Script | Backend | Sim pytest | Device / CI | Notes |
| --- | --- | --- | --- | --- |
| `broadcast.py` | real | yes (needs ttnn) | yes | |
| `broadcast_demo.py` | sim | yes | yes | `from sim import ttl, ttnn`; runs in sim pytest and in the hardware batch. |
| `general_broadcast.py` | real | yes (needs ttnn) | yes | |
| `eltwise_add.py` | real | yes | yes | |
| `eltwise_add_3d.py` | real | yes | yes | |
| `errors/eltwise_add_error.py` | real | negative test (expect failure) | skip | Demonstrates a bad copy; must not exit 0. |
| `eltwise_pipe.py` | real | yes | yes | |
| `eltwise_pipe_core3.py` | real | yes | yes | |
| `matmul.py` | sim | yes | skip | Sim-backed; not in hardware batch until migrated (unlike `broadcast_demo.py`). |
| `matmul_explicit_acc.py` | real | yes | yes | |
| `singlecore_matmul.py` | real | yes | yes | |
| `multicore_matmul.py` | real | yes | yes | |
| `matmul_1d.py` | real | yes | skip | Opted out of default hardware batch (see file header). |
| `matmul_1d_mcast.py` | real | yes (fair scheduler skipped) | skip | Fair scheduler times out in sim (see pytest skip). |
| `eltwise_1d_broadcast.py` | sim | yes | skip | Simulator-backed; hardware batch uses real stack only. |
| `errors/copy_lock_error.py` | real | negative test (expect failure) | skip | Demonstrates copy-lock error; must not exit 0. |
| `test_transformer_block.py` | real | no (not in `test_example_cli`) | skip | Large demo; run manually on device. |
| `tutorial/single_core_single_tile_block.py` | real | yes (needs ttnn) | yes | |
| `tutorial/single_core_multitile_block.py` | real | yes (needs ttnn) | yes | |
| `tutorial/single_core_broadcast_single_tile_block.py` | real | yes (needs ttnn) | yes | |
| `tutorial/single_core_broadcast_multitile_blocks.py` | real | yes (needs ttnn) | yes | |
| `tutorial/multicore.py` | real | yes (needs ttnn) | yes | |
| `tutorial/multicore_grid_auto.py` | real | yes (needs ttnn) | yes | |
| `tutorial/ttnn_base.py` | real | yes (needs ttnn) | no | No `@ttl.operation`; not discovered by compile-and-run. |

`requires_ttnn` in `test_examples.py` skips rows that need a real `ttnn` import for golden checks when `ttnn` is missing from the environment.

## Negative examples (sim only)

These are tested explicitly to **fail** with a known diagnostic:

- `errors/eltwise_add_error.py` — shape mismatch on copy.
- `errors/copy_lock_error.py` — copy-lock / `tx.wait()` ordering.

They are tagged `skip-compiler` so the hardware batch does not treat failure as a regression.

## Migrating sim-only examples

Files that use `from sim import ttl, ttnn` run under `ttlang-sim` and pytest with the simulator shadowing real `ttnn`. `broadcast_demo.py` is kept on that import style and is included in the hardware compile-and-run batch. Other sim-backed examples (for example `matmul.py`) may still be migrated to match `examples/broadcast.py` (`import ttl`, `ttnn.open_device`, `ttnn.from_torch(..., device=..., memory_config=...)`, `ttnn.close_device` in `finally`) before dropping `skip-compiler`.

## Metal examples

`examples/metal_examples/**` are covered by `test_metal_example_cli` in `test/sim/test_examples.py` (not by the hardware compile-and-run script).
