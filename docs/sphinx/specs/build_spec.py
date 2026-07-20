# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate TTLangSpecification.md from its template and externalized examples.

The specification is authored as ``TTLangSpecification.externalized.md``, a
Markdown file whose code examples are replaced by include directives of the
form::

    <!-- @spec:example <file> -->

Each directive names a Python file under ``examples/spec/``. That file is a
standalone example; the region to embed in the specification is delimited by
marker comments::

    # spec:begin
    ...code included in the specification...
    # spec:end

The embedded region is dedented, so an example may nest the marked lines inside
a function while the specification shows them at the left margin.

A single directive may match several ``spec:begin`` / ``spec:end`` sections in
one file. The sections are concatenated first, with no separating line, and the
result is dedented as a whole, so an example can skip over intervening
scaffolding and embed only the relevant fragments while their shared
indentation is stripped consistently.

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


def _repo_root() -> Path:
    """Nearest ancestor containing pyproject.toml."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SystemExit("build_spec.py: no pyproject.toml found in any ancestor")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples" / "spec"
TEMPLATE = SCRIPT_DIR / "TTLangSpecification.externalized.md"
OUTPUT = SCRIPT_DIR / "TTLangSpecification.md"

DIRECTIVE = re.compile(r"^<!-- @spec:example (?P<file>\S+) -->[ \t]*$")

BEGIN = "# spec:begin"
END = "# spec:end"


def extract_region(example: Path) -> str:
    """Return the dedented lines between the spec markers in ``source``.

    When ``source`` contains several ``# spec:begin`` / ``# spec:end`` sections,
    they are concatenated (with no separating line) and the result is dedented
    as a whole.
    """
    source = example.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    regions: list[str] = []
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == BEGIN:
            if start is not None:
                raise SystemExit(
                    f"Nested {BEGIN!r} marker before its {END!r} in {example}"
                )
            start = i
        elif stripped == END:
            if start is None:
                raise SystemExit(
                    f"{END!r} marker without a preceding {BEGIN!r} in {example}"
                )
            regions.append("".join(lines[start + 1 : i]))
            start = None

    if start is not None:
        raise SystemExit(f"{BEGIN!r} marker without a closing {END!r} in {example}")
    if not regions:
        raise SystemExit(f"Could not find {BEGIN}/{END} markers in {example}")

    return textwrap.dedent("".join(regions))


def render(template_text: str) -> str:
    """Expand every include directive in ``template_text`` into a code block."""
    out: list[str] = []
    for line in template_text.splitlines(keepends=True):
        match = DIRECTIVE.match(line.rstrip("\n"))
        if match is None:
            if "@spec:" in line:
                raise SystemExit(f"Malformed directive: {line.strip()}")
            out.append(line)
            continue

        example = EXAMPLES_DIR / match["file"]
        if not example.is_file():
            raise SystemExit(f"Example file not found: {example}")

        region = extract_region(example)
        if not region:
            raise SystemExit(f"Empty region in {example}")

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
