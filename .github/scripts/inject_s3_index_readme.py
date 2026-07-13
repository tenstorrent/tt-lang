#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Inject the S3-index README into a slash-key or root index.

Direct S3 wheel-directory publishes regenerate slash-key listings. This restores
the README above anchors without changing pip resolution; repeated runs replace
the existing sentinel-delimited block.

Usage:
  inject_s3_index_readme.py [--create-from-dist <dist_dir>] <readme.md> <index.html>
"""

import argparse
import html
import re
import sys
from pathlib import Path

START = "<!-- ttlang-s3-readme:start -->"
END = "<!-- ttlang-s3-readme:end -->"


def render_markdown(text: str) -> str:
    """Render Markdown to an HTML fragment, falling back to escaped text.

    The fallback keeps publishing independent of the optional `markdown` package.
    """
    try:
        import markdown
    except ImportError:
        return "<pre>\n" + html.escape(text) + "\n</pre>"
    return markdown.markdown(text, extensions=["fenced_code"])


def build_block(readme_md: str) -> str:
    fragment = render_markdown(readme_md)
    return f'{START}\n<section id="ttlang-s3-readme">\n{fragment}\n</section>\n{END}\n'


def build_standalone_readme(readme_md: str) -> str:
    fragment = render_markdown(readme_md)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="UTF-8">\n'
        "<title>tt-lang Tenstorrent S3 PyPI index</title>\n"
        "</head>\n"
        "<body>\n"
        '<section id="ttlang-s3-readme">\n'
        f"{fragment}\n"
        "</section>\n"
        "</body>\n"
        "</html>\n"
    )


def normalize_project_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution_name).lower()


def project_names_from_wheels(dist_dir: Path) -> list[str]:
    names = {
        normalize_project_name(wheel.name.split("-", 1)[0])
        for wheel in dist_dir.glob("*.whl")
        if "-" in wheel.name
    }
    if not names:
        raise ValueError(f"No wheel files found under {dist_dir}")
    return sorted(names)


def build_root_index(dist_dir: Path) -> str:
    anchors = "\n".join(
        f'<a href="{html.escape(project_name)}/">{html.escape(project_name)}</a>'
        for project_name in project_names_from_wheels(dist_dir)
    )
    return f"<!DOCTYPE html>\n<html>\n<body>\n{anchors}\n</body>\n</html>\n"


def inject(index_html: str, block: str) -> str:
    """Return index_html with block placed above the anchor links.

    Existing injected content is removed first. Inserting after <body> keeps
    package anchors below the README.
    """
    existing = re.compile(re.escape(START) + ".*?" + re.escape(END) + r"\n?", re.DOTALL)
    index_html = existing.sub("", index_html)

    body = re.search(r"<body[^>]*>", index_html, re.IGNORECASE)
    if body:
        cut = body.end()
        return index_html[:cut] + "\n" + block + index_html[cut:]
    return block + index_html


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Inject the tt-lang S3 README into an index.html file."
    )
    parser.add_argument(
        "--create-from-dist",
        metavar="DIST_DIR",
        help=(
            "Create a minimal root package index from wheel files when "
            "index.html is absent."
        ),
    )
    parser.add_argument(
        "--render-readme-html",
        action="store_true",
        help="Render the README as a standalone HTML file at the index path.",
    )
    parser.add_argument("readme")
    parser.add_argument("index")
    args = parser.parse_args(argv[1:])

    readme_path = Path(args.readme)
    index_path = Path(args.index)

    with open(readme_path, encoding="utf-8") as handle:
        readme_md = handle.read()
    if args.render_readme_html:
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write(build_standalone_readme(readme_md))
        return 0

    if index_path.exists():
        with open(index_path, encoding="utf-8") as handle:
            index_html = handle.read()
    elif args.create_from_dist:
        try:
            index_html = build_root_index(Path(args.create_from_dist))
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
    else:
        print(f"Index file does not exist: {index_path}", file=sys.stderr)
        return 1

    result = inject(index_html, build_block(readme_md))

    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
