# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source-size counts distinguish comments/docstrings from string contents."""

from benchmarks.all_gather_minimal_matmul.source_size import cpp_sloc, python_sloc


def test_python_sloc_excludes_docstrings_and_comments():
    source = '''"""Module documentation."""
# Comment
def operation():
    """Function
    documentation.
    """
    value = "# string contents"  # Comment
    return value
'''
    assert python_sloc(source) == 3


def test_python_sloc_counts_multiline_values():
    assert python_sloc("value = (\n    1,\n    # Comment\n    2,\n)\n") == 4


def test_cpp_sloc_preserves_strings():
    source = """// Comment
const char* value = "https://example.org";
/* Multiple
   comment lines */
return value; // Comment
"""
    assert cpp_sloc(source) == 2


def test_cpp_sloc_preserves_raw_strings():
    assert cpp_sloc('const char* value = R"tag(// payload\n/* payload */)tag";\n') == 2
