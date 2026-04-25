# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CompilerOptions parsing, merging, and argv extraction."""

import sys
from unittest import mock

import pytest

import ttl.compiler_options as _co
from ttl.compiler_options import CompilerOptions


class TestDefaults:
    def test_defaults(self):
        opts = CompilerOptions()
        assert opts.maximize_dst is True
        assert opts.enable_fpu_binary_ops is True
        assert opts.subblock_sync is False
        assert opts._explicit == frozenset()

    def test_frozen(self):
        opts = CompilerOptions()
        with pytest.raises(AttributeError):
            opts.maximize_dst = False  # type: ignore[misc]


class TestFromString:
    def test_none_returns_defaults(self):
        opts = CompilerOptions.from_string(None)
        assert opts == CompilerOptions()

    def test_empty_returns_defaults(self):
        opts = CompilerOptions.from_string("")
        assert opts == CompilerOptions()

    def test_disable_maximize_dst(self):
        opts = CompilerOptions.from_string("--no-ttl-maximize-dst")
        assert opts.maximize_dst is False
        assert opts.enable_fpu_binary_ops is True
        assert "maximize_dst" in opts._explicit

    def test_disable_fpu(self):
        opts = CompilerOptions.from_string("--no-ttl-fpu-binary-ops")
        assert opts.enable_fpu_binary_ops is False
        assert "enable_fpu_binary_ops" in opts._explicit

    def test_enable_subblock_sync(self):
        opts = CompilerOptions.from_string("--ttl-subblock-sync")
        assert opts.subblock_sync is True
        assert "subblock_sync" in opts._explicit

    def test_enable_flags_explicitly(self):
        opts = CompilerOptions.from_string("--ttl-maximize-dst --ttl-fpu-binary-ops")
        assert opts.maximize_dst is True
        assert opts.enable_fpu_binary_ops is True
        assert opts._explicit == frozenset({"maximize_dst", "enable_fpu_binary_ops"})

    def test_contradictory_last_wins(self):
        opts = CompilerOptions.from_string("--no-ttl-maximize-dst --ttl-maximize-dst")
        assert opts.maximize_dst is True

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel option"):
            CompilerOptions.from_string("--nonexistent-flag")

    def test_multiple_flags(self):
        opts = CompilerOptions.from_string(
            "--no-ttl-maximize-dst --no-ttl-fpu-binary-ops"
        )
        assert opts.maximize_dst is False
        assert opts.enable_fpu_binary_ops is False


class TestMerge:
    def test_override_replaces_base(self):
        base = CompilerOptions.from_string("--ttl-maximize-dst")
        override = CompilerOptions.from_string("--no-ttl-maximize-dst")
        result = base.merge(override)
        assert result.maximize_dst is False

    def test_unset_override_preserves_base(self):
        base = CompilerOptions.from_string("--no-ttl-maximize-dst")
        override = CompilerOptions()  # nothing explicit
        result = base.merge(override)
        assert result.maximize_dst is False

    def test_both_set_same_field(self):
        base = CompilerOptions.from_string("--no-ttl-fpu-binary-ops")
        override = CompilerOptions.from_string("--ttl-fpu-binary-ops")
        result = base.merge(override)
        assert result.enable_fpu_binary_ops is True

    def test_merge_tracks_combined_explicit(self):
        base = CompilerOptions.from_string("--no-ttl-maximize-dst")
        override = CompilerOptions.from_string("--no-ttl-fpu-binary-ops")
        result = base.merge(override)
        assert result._explicit == frozenset({"maximize_dst", "enable_fpu_binary_ops"})

    def test_merge_defaults_onto_defaults(self):
        result = CompilerOptions().merge(CompilerOptions())
        assert result == CompilerOptions()
        assert result._explicit == frozenset()


class TestFromArgv:
    @pytest.fixture(autouse=True)
    def _reset_argv_cache(self):
        """Clear the from_argv() cache so each test parses fresh."""
        _co._argv_result = None

    def test_extracts_known_flags(self):
        with mock.patch.object(
            sys, "argv", ["script.py", "--no-ttl-maximize-dst", "some_file.py"]
        ):
            opts = CompilerOptions.from_argv()
        assert opts.maximize_dst is False
        assert opts.enable_fpu_binary_ops is True  # default

    def test_ignores_unknown_flags(self):
        with mock.patch.object(
            sys,
            "argv",
            ["script.py", "--unknown-flag", "-v", "--no-ttl-fpu-binary-ops"],
        ):
            opts = CompilerOptions.from_argv()
        assert opts.enable_fpu_binary_ops is False

    def test_no_flags_returns_defaults(self):
        with mock.patch.object(sys, "argv", ["script.py"]):
            opts = CompilerOptions.from_argv()
        assert opts == CompilerOptions()

    def test_help_exits(self):
        with mock.patch.object(sys, "argv", ["my_kernel.py", "--ttl-help"]):
            with pytest.raises(SystemExit) as exc_info:
                CompilerOptions.from_argv()
            assert exc_info.value.code == 0

    def test_help_with_other_flags(self):
        """--ttl-help triggers exit regardless of other flags."""
        with mock.patch.object(
            sys, "argv", ["pytest", "--ttl-help", "--no-ttl-maximize-dst"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                CompilerOptions.from_argv()
            assert exc_info.value.code == 0


class TestEquality:
    def test_same_values_equal_regardless_of_explicit(self):
        a = CompilerOptions()
        b = CompilerOptions.from_string("--ttl-maximize-dst --ttl-fpu-binary-ops")
        assert a == b

    def test_different_values_not_equal(self):
        a = CompilerOptions()
        b = CompilerOptions.from_string("--no-ttl-maximize-dst")
        assert a != b

    def test_hash_consistent_with_equality(self):
        a = CompilerOptions()
        b = CompilerOptions.from_string("--ttl-maximize-dst --ttl-fpu-binary-ops")
        assert hash(a) == hash(b)
