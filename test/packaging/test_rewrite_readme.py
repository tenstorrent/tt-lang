# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the README-rewrite helpers used by both wheel setup scripts."""

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "packaging"))

from rewrite_readme import (  # noqa: E402
    absolutize_readme_images,
    ref_for_version,
)


# ---------------------------------------------------------------------------
# ref_for_version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.2.3", "v1.2.3"),
        ("0.0.1", "v0.0.1"),
        ("10.20.30", "v10.20.30"),
    ],
)
def test_ref_for_version_release(version, expected):
    assert ref_for_version(version) == expected


@pytest.mark.parametrize(
    "version",
    [
        # dev/local segments produced by get_version_from_git
        "1.2.3.dev5",
        "1.2.3.dev0",
        "1.2.3+local",
        "1.2.3.dev2+local",
        # PEP 440 pre/post-release identifiers — not in this repo's tag scheme
        "1.2.3rc1",
        "1.2.3a1",
        "1.2.3b2",
        "1.2.3.post1",
        # Malformed TTLANG_VERSION_OVERRIDE overrides
        "1.2.3-rc1",
        "foo",
        "",
        "1",
        "1.2",
        "v1.2.3",
        "1.2.3.4",
        "1.2.3 ",
    ],
)
def test_ref_for_version_non_release_falls_back_to_main(version):
    assert ref_for_version(version) == "main"


# ---------------------------------------------------------------------------
# absolutize_readme_images
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs" / "img").mkdir(parents=True)
    (tmp_path / "docs" / "img" / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "docs" / "img" / "b.png").write_bytes(b"\x89PNG")
    return tmp_path


_BASE = "https://raw.githubusercontent.com/tenstorrent/tt-lang/v1.2.3"


def test_rewrites_html_img_relative_src(repo):
    text = '<img alt="logo" src="docs/img/a.png" height="100">'
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == f'<img alt="logo" src="{_BASE}/docs/img/a.png" height="100">'


def test_rewrites_markdown_image_relative_path(repo):
    text = "![alt text](docs/img/a.png)"
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == f"![alt text]({_BASE}/docs/img/a.png)"


def test_preserves_https_urls(repo):
    text = (
        '<img src="https://img.shields.io/badge/x.svg">\n'
        "![badge](https://example.com/y.png)\n"
    )
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == text


def test_preserves_absolute_path_and_data_uri(repo):
    text = '<img src="/static/abs.png">\n' '<img src="data:image/png;base64,AAAA">\n'
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == text


def test_handles_uppercase_img_tag(repo):
    text = '<IMG SRC="docs/img/a.png">'
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == f'<IMG SRC="{_BASE}/docs/img/a.png">'


def test_preserves_markdown_image_title(repo):
    text = '![alt](docs/img/a.png "caption")'
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == f'![alt]({_BASE}/docs/img/a.png "caption")'


def test_does_not_match_plain_markdown_link(repo):
    # A link (no leading '!') with a relative path must remain untouched even
    # if the target does not exist on disk.
    text = "[click](docs/img/does_not_exist.html)"
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == text


def test_does_not_match_image_tag_variant(repo):
    # <image> is SVG, not HTML <img>; leave it alone even if src is relative.
    text = '<image src="docs/img/does_not_exist.png" />'
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert out == text


def test_missing_file_aborts_and_lists_all_paths(repo):
    text = (
        '<img src="docs/img/a.png">\n'
        '<img src="docs/img/missing_html.png">\n'
        "![alt](docs/img/missing_md.png)\n"
    )
    with pytest.raises(FileNotFoundError) as info:
        absolutize_readme_images(text, "v1.2.3", repo)
    message = str(info.value)
    assert "missing_html.png" in message
    assert "missing_md.png" in message
    # Files that do exist must not appear in the error list.
    assert "docs/img/a.png" not in message.replace("missing_html.png", "").replace(
        "missing_md.png", ""
    )


def test_rewrites_multiple_images_in_one_document(repo):
    text = (
        "<picture>\n"
        '  <img alt="A" src="docs/img/a.png" height="220">\n'
        "</picture>\n"
        "\n"
        "Body text.\n"
        "\n"
        "![B](docs/img/b.png)\n"
    )
    out = absolutize_readme_images(text, "v1.2.3", repo)
    assert f'src="{_BASE}/docs/img/a.png"' in out
    assert f"({_BASE}/docs/img/b.png)" in out


def test_ref_is_threaded_through(repo):
    text = '<img src="docs/img/a.png">'
    out = absolutize_readme_images(text, "main", repo)
    assert (
        "https://raw.githubusercontent.com/tenstorrent/tt-lang/main/docs/img/a.png"
        in out
    )
