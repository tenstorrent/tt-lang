#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import os
import string
import subprocess
import sys
from pathlib import Path


_DEFAULT_SIGNATURES = {
    5: frozenset(
        {
            "88a5c49d8cb7f295df955627440db159fecd962f91d56017b451fc9d6facd79f",
        }
    ),
}
_ZERO_OBJECT_IDS = frozenset({"0" * 40, "0" * 64})


class CheckFailure(Exception):
    pass


def run_git(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        command = " ".join(["git", *arguments])
        raise CheckFailure(f"command failed: {command}")
    return result


def load_signatures() -> dict[int, frozenset[str]]:
    signatures = {
        content_length: set(content_hashes)
        for content_length, content_hashes in _DEFAULT_SIGNATURES.items()
    }
    configured_signatures = os.environ.get("TTLANG_PUBLIC_CONTENT_SIGNATURES", "")
    for entry in filter(None, configured_signatures.split(",")):
        content_length_text, separator, content_hash = entry.partition(":")
        if (
            not separator
            or not content_length_text.isdecimal()
            or int(content_length_text) < 1
            or len(content_hash) != 64
            or any(character not in string.hexdigits for character in content_hash)
        ):
            raise CheckFailure("invalid public-content signature configuration")
        signatures.setdefault(int(content_length_text), set()).add(content_hash.lower())
    return {
        content_length: frozenset(content_hashes)
        for content_length, content_hashes in signatures.items()
    }


def contains_restricted_content(
    content: bytes, signatures: dict[int, frozenset[str]]
) -> bool:
    normalized_content = content.lower()
    for content_length, content_hashes in signatures.items():
        final_offset = len(normalized_content) - content_length + 1
        for offset in range(max(final_offset, 0)):
            content_hash = hashlib.sha256(
                normalized_content[offset : offset + content_length]
            ).hexdigest()
            if content_hash in content_hashes:
                return True
    return False


def check_content(
    description: str, content: bytes, signatures: dict[int, frozenset[str]]
) -> None:
    if contains_restricted_content(content, signatures):
        raise CheckFailure(f"{description} contains restricted public content")


def ref_exists(ref: str) -> bool:
    result = run_git(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], check=False
    )
    return result.returncode == 0


def current_branch_name() -> str:
    branch_name = os.environ.get("GITHUB_HEAD_REF", "")
    if branch_name:
        return branch_name

    result = run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
    if result.returncode == 0:
        return result.stdout.decode().strip()
    return os.environ.get("GITHUB_REF_NAME", "")


def resolve_diff_base() -> str:
    configured_base = os.environ.get("TTLANG_PUBLIC_DIFF_BASE", "")
    if configured_base:
        return configured_base

    github_base_branch = os.environ.get("GITHUB_BASE_REF", "")
    if github_base_branch:
        return f"refs/remotes/origin/{github_base_branch}"
    return "refs/remotes/origin/main"


def check_branch_diff(target_ref: str, signatures: dict[int, frozenset[str]]) -> None:
    base_ref = resolve_diff_base()
    if not ref_exists(base_ref):
        raise CheckFailure(f"cannot resolve required diff base {base_ref}")

    branch_diff = run_git(
        ["diff", "--no-ext-diff", "--no-color", f"{base_ref}...{target_ref}"]
    ).stdout
    check_content("branch diff", branch_diff, signatures)


def check_change(signatures: dict[int, frozenset[str]]) -> None:
    check_content("branch name", current_branch_name().encode(), signatures)
    check_branch_diff("HEAD", signatures)
    staged_diff = run_git(["diff", "--cached", "--no-ext-diff", "--no-color"]).stdout
    check_content("staged diff", staged_diff, signatures)


def check_commit_message(
    commit_message_file: str, signatures: dict[int, frozenset[str]]
) -> None:
    check_content("branch name", current_branch_name().encode(), signatures)
    git_directory = Path(
        run_git(["rev-parse", "--absolute-git-dir"]).stdout.decode().strip()
    ).resolve(strict=True)
    message_file = Path(commit_message_file).resolve(strict=True)
    if not message_file.is_relative_to(git_directory):
        raise CheckFailure("commit message file is outside Git metadata")
    check_content("commit message", message_file.read_bytes(), signatures)


def check_push(signatures: dict[int, frozenset[str]]) -> None:
    target_ref = os.environ.get("PRE_COMMIT_TO_REF", "")
    if target_ref in _ZERO_OBJECT_IDS:
        return

    if not target_ref:
        target_ref = os.environ.get("PRE_COMMIT_LOCAL_BRANCH", "") or "HEAD"
    if not ref_exists(target_ref):
        raise CheckFailure(f"cannot resolve pushed target ref {target_ref}")

    for description, branch_name in (
        (
            "local branch name",
            os.environ.get("PRE_COMMIT_LOCAL_BRANCH", "") or current_branch_name(),
        ),
        ("remote branch name", os.environ.get("PRE_COMMIT_REMOTE_BRANCH", "")),
    ):
        check_content(description, branch_name.encode(), signatures)

    check_branch_diff(target_ref, signatures)

    source_ref = os.environ.get("PRE_COMMIT_FROM_REF", "")
    if not source_ref or source_ref in _ZERO_OBJECT_IDS or not ref_exists(source_ref):
        source_ref = resolve_diff_base()
    if not ref_exists(source_ref):
        raise CheckFailure(f"cannot resolve pushed source ref {source_ref}")

    commit_messages = run_git(
        ["log", "--format=%B%x00", f"{source_ref}..{target_ref}"]
    ).stdout
    check_content("pushed commit-message data", commit_messages, signatures)

    commit_patches = run_git(
        [
            "log",
            "--format=",
            "--patch",
            "--diff-merges=separate",
            "--no-ext-diff",
            "--no-color",
            f"{source_ref}..{target_ref}",
        ]
    ).stdout
    check_content("pushed commit-patch data", commit_patches, signatures)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("change", "commit-message", "push"))
    parser.add_argument("commit_message_file", nargs="?")
    arguments = parser.parse_args()
    if (arguments.mode == "commit-message") != bool(arguments.commit_message_file):
        parser.error("commit-message mode requires exactly one message file")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    try:
        signatures = load_signatures()
        if arguments.mode == "change":
            check_change(signatures)
        elif arguments.mode == "commit-message":
            check_commit_message(arguments.commit_message_file, signatures)
        else:
            check_push(signatures)
    except (CheckFailure, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
