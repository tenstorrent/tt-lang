# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Count maintained operation source, excluding comments, docstrings and blank lines."""

import argparse
import ast
import io
from pathlib import Path
import re
import subprocess
import tokenize

NATIVE_REVISION = "ea042c4ad6237678103cd7cbceb346e060f0f9a3"
NATIVE_DIRECTORY = (
    "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async"
)
CPP_TOKENS = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|//[^\n]*|/\*[\s\S]*?\*/"
)


def python_sloc(source):
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and ast.get_docstring(node, clean=False) is not None
        ):
            expression = node.body[0]
            docstrings.add((expression.lineno, expression.col_offset))
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    source_lines = source.splitlines()
    code_lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored or (
            token.type == tokenize.STRING and token.start in docstrings
        ):
            continue
        code_lines.update(
            line
            for line in range(token.start[0], token.end[0] + 1)
            if source_lines[line - 1].strip()
        )
    return len(code_lines)


def cpp_sloc(source):
    def strip_comment(match):
        token = match.group()
        return "\n" * token.count("\n") if token.startswith(("//", "/*")) else token

    return sum(
        bool(line.strip())
        for line in CPP_TOKENS.sub(strip_comment, source).splitlines()
    )


def git(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    )


def measure(ttlang_root, native_root, native_revision=NATIVE_REVISION):
    example = ttlang_root / "examples/all_gather_minimal_matmul"
    shared = ["config.py", "collectives.py"]
    versions = {
        "V1 per-row": [*shared, "operation.py", "per_row_all_gather/operation.py"],
        "V2 two-worker ring": [*shared, "operation.py", "two_worker_ring/operation.py"],
        "V3 replicated": [*shared, "replicated/operation.py"],
        "V4 dedicated communication": [
            *shared,
            "operation.py",
            "dedicated_communication/operation.py",
        ],
    }
    native_files = [
        filename
        for filename in git(
            native_root,
            "ls-tree",
            "-r",
            "--name-only",
            native_revision,
            "--",
            NATIVE_DIRECTORY,
        ).splitlines()
        if filename.endswith((".cpp", ".hpp")) and "nanobind" not in filename
    ]
    if not native_files:
        raise ValueError("native revision contains no operation sources")
    native_counts = {
        filename: cpp_sloc(git(native_root, "show", f"{native_revision}:{filename}"))
        for filename in native_files
    }
    python_counts = {
        filename: python_sloc((example / filename).read_text())
        for filename in sorted(
            {filename for files in versions.values() for filename in files}
        )
    }
    return versions, python_counts, native_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-metal", type=Path, required=True)
    parser.add_argument("--native-revision", default=NATIVE_REVISION)
    arguments = parser.parse_args()
    ttlang_root = Path(__file__).resolve().parents[2]
    versions, python_counts, native_counts = measure(
        ttlang_root, arguments.tt_metal, arguments.native_revision
    )
    native_total = sum(native_counts.values())
    print(
        f"TT-Lang: {git(ttlang_root, 'rev-parse', 'HEAD').strip()}; native: {arguments.native_revision}"
    )
    print("Version | TT-Lang SLOC | Native SLOC | TT-Lang/native")
    for version, files in versions.items():
        total = sum(python_counts[filename] for filename in files)
        print(f"{version} | {total} | {native_total} | {total / native_total:.3f}")
    print("\nCounted files:")
    for filename, count in {**python_counts, **native_counts}.items():
        print(f"{count:5} {filename}")


if __name__ == "__main__":
    main()
