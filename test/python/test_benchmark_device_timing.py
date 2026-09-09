# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check benchmark selection and aggregation of TT-Metal processor results."""

import pytest

from benchmarks.device_timing import latest_kernel_duration, read_device_profile


def processed_operation(cycles, run_id):
    return {
        "analysis": {
            "device_kernel_duration": {
                "series": [
                    {
                        "duration_cycles": cycles,
                        "duration_type": (
                            {"run_host_id": run_id},
                            {"run_host_id": run_id},
                        ),
                    }
                ]
            }
        }
    }


@pytest.fixture
def profiler_data(monkeypatch):
    from tracy import process_device_log

    data = {
        "deviceInfo": {"freq": 1350},
        "devices": {
            device_id: {
                "cores": {
                    "DEVICE": {
                        "riscs": {
                            "TENSIX": {
                                "ops": [
                                    processed_operation(90000, 1),
                                    processed_operation(cycles, 10 + device_id),
                                ]
                            }
                        }
                    }
                }
            }
            for device_id, cycles in ((1, 13500), (2, 27000), (3, 100000))
        },
    }

    def import_profile(setup):
        assert set(setup.timerAnalysis) == {"device_kernel_duration"}
        assert setup.timerAnalysis["device_kernel_duration"]["type"] == "op_first_last"
        return data

    monkeypatch.setattr(process_device_log, "import_log_run_stats", import_profile)
    return data


def test_latest_participant_programs(profiler_data):
    result = latest_kernel_duration("unused.csv", [1, 2])
    assert result["cycles"] == 27000
    assert result["us"] == 20
    assert result["per_device"] == {
        "1": {"cycles": 13500, "run_host_id": 11},
        "2": {"cycles": 27000, "run_host_id": 12},
    }


def test_invalid_frequency(profiler_data):
    profiler_data["deviceInfo"]["freq"] = 0
    with pytest.raises(ValueError, match="invalid profiler clock"):
        latest_kernel_duration("unused.csv", [1, 2])


def test_missing_participant(profiler_data):
    with pytest.raises(KeyError):
        latest_kernel_duration("unused.csv", [0, 1])


def test_missing_launch(profiler_data):
    profiler_data["devices"][1]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"] = []
    with pytest.raises(ValueError, match="no profiled operations"):
        latest_kernel_duration("unused.csv", [1, 2])


def test_mismatched_launch_boundaries(profiler_data):
    operation = profiler_data["devices"][1]["cores"]["DEVICE"]["riscs"]["TENSIX"][
        "ops"
    ][-1]
    operation["analysis"]["device_kernel_duration"]["series"][0]["duration_type"][1][
        "run_host_id"
    ] = 99
    with pytest.raises(ValueError, match="different program launches"):
        latest_kernel_duration("unused.csv", [1, 2])


def test_profiler_dumps_are_preserved(monkeypatch, tmp_path):
    import ttnn

    monkeypatch.setenv("TT_METAL_PROFILER_DIR", str(tmp_path))
    log = tmp_path / ".logs/profile_log_device.csv"
    log.parent.mkdir()
    monkeypatch.setattr(
        ttnn, "ReadDeviceProfiler", lambda device: log.write_text("profiler dump\n")
    )
    first_dump = read_device_profile(None)
    second_dump = read_device_profile(None)
    assert first_dump != second_dump
    assert first_dump.read_text() == second_dump.read_text() == "profiler dump\n"
    assert not log.exists()
