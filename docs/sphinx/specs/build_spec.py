# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate TTLangSpecification.md from its template and externalized examples.

The specification is authored as ``TTLangSpecification.externalized.md``, a
Markdown file whose code examples are replaced by include directives of the
form::

    <!-- @spec:example <file> -->
    <!-- @spec:example <file>:<tag> -->

Each directive names a Python file under ``examples/spec/``. That file is a
standalone example; the region to embed in the specification is delimited by
marker comments::

    # spec:begin
    ...code included in the specification...
    # spec:end

A file may contain several named regions, selected by ``:<tag>`` on both the
directive and the markers (``# spec:begin <tag>`` / ``# spec:end <tag>``). The
embedded region is dedented, so an example may nest the marked lines inside a
function while the specification shows them at the left margin.

Usage::

    python docs/sphinx/specs/build_spec.py            # write the specification
    python docs/sphinx/specs/build_spec.py --check    # verify it is up to date

``--check`` regenerates the specification in memory and exits non-zero (without
writing) if it differs from the committed file, for use in CI.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "spec"
TEMPLATE = SCRIPT_DIR / "TTLangSpecification.externalized.md"
OUTPUT = SCRIPT_DIR / "TTLangSpecification.md"

DIRECTIVE = re.compile(
    r"^<!-- @spec:example (?P<file>[^\s:]+)(?::(?P<tag>\S+))? -->[ \t]*$"
)


def extract_region(source: str, tag: str | None) -> str:
    """Return the dedented lines between the spec markers in ``source``.

    ``tag`` selects a named region; ``None`` selects the plain
    ``# spec:begin`` / ``# spec:end`` pair.
    """
    begin = "# spec:begin" if tag is None else f"# spec:begin {tag}"
    end = "# spec:end" if tag is None else f"# spec:end {tag}"

    lines = source.splitlines(keepends=True)
    start = end_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == begin and start is None:
            start = i
        elif stripped == end and start is not None:
            end_idx = i
            break

    if start is None or end_idx is None:
        marker = "spec:begin/spec:end" if tag is None else f"spec:begin/end '{tag}'"
        raise SystemExit(f"Could not find {marker} markers")

    return textwrap.dedent("".join(lines[start + 1 : end_idx]))


def render(template_text: str) -> str:
    """Expand every include directive in ``template_text`` into a code block."""
    out: list[str] = []
    for line in template_text.splitlines(keepends=True):
        match = DIRECTIVE.match(line.rstrip("\n"))
        if match is None:
            out.append(line)
            continue

        example = EXAMPLES_DIR / match["file"]
        if not example.is_file():
            raise SystemExit(f"Example file not found: {example}")

        region = extract_region(example.read_text(encoding="utf-8"), match["tag"])
        if not region.endswith("\n"):
            region += "\n"
        out.append("```py\n")
        out.append(region)
        out.append("```\n")

    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the specification is up to date without writing it",
    )
    args = parser.parse_args()

    generated = render(TEMPLATE.read_text(encoding="utf-8"))

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != generated:
            print(
                f"{OUTPUT.name} is out of date; regenerate with "
                f"`python {Path(__file__).relative_to(REPO_ROOT)}`",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.name} is up to date.")
        return 0

    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
