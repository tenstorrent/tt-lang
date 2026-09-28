#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <tt-metal-source> <new-runtime-directory>" >&2
    exit 2
fi

SOURCE="$(cd "$1" && pwd -P)"
if [ -e "$2" ] || [ -L "$2" ]; then
    echo "Runtime destination must not already exist: $2" >&2
    exit 1
fi
DESTINATION="$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")"
case "$DESTINATION/" in
    "$SOURCE/"*)
        echo "Runtime destination must be outside the source tree." >&2
        exit 1
        ;;
esac

for required in runtime/hw runtime/sfpi tt_metal tt_stl ttnn/cpp ttnn/ttnn \
    build_emule/generated/tt_metal/impl/version.hpp \
    build_emule/tt_metal/libtt_metal.so build_emule/tt_stl/libtt_stl.so \
    build_emule/ttnn/_ttnn.so build_emule/ttnn/_ttnncpp.so; do
    if [ ! -e "$SOURCE/$required" ]; then
        echo "Required emulator runtime artifact is missing: $required" >&2
        exit 1
    fi
done

mkdir "$DESTINATION"
# The native paths are embedded in Metal's JIT and shared-library search paths.
(cd "$SOURCE" && tar \
    --exclude=.git --exclude=__pycache__ --exclude=compile_commands.json \
    --exclude='*/tests' --exclude='*/docs' --exclude='ttnn/ttnn/*.so' \
    -cf - runtime tt_metal tt_stl ttnn/cpp ttnn/ttnn LICENSE NOTICE) |
    (cd "$DESTINATION" && tar -xf -)

while IFS= read -r -d '' artifact; do
    relative="${artifact#"$SOURCE/"}"
    mkdir -p "$DESTINATION/$(dirname "$relative")"
    cp -pP "$artifact" "$DESTINATION/$relative"
done < <(find "$SOURCE/build_emule" \
    \( -name CMakeFiles -o -name tt-metal-cache \) -prune -o \
    \( -type f -o -type l \) \
    \( -name '*.so' -o -name '*.so.*' -o -name '*.h' -o -name '*.hpp' \
       -o -name '*.inc' -o -name '*.cc' -o -name '*.cpp' \) -print0)

ln -s ../../build_emule/ttnn/_ttnn.so "$DESTINATION/ttnn/ttnn/_ttnn.so"

# Dependency notices remain available after their build-only sources are omitted.
for dependency_tree in .cpmcache third_party; do
    if [ ! -d "$SOURCE/$dependency_tree" ]; then
        continue
    fi
    while IFS= read -r -d '' notice; do
        case "$(realpath "$notice")" in
            "$SOURCE"/*) ;;
            *)
                echo "Dependency notice escapes the source tree: $notice" >&2
                exit 1
                ;;
        esac
        relative="${notice#"$SOURCE/"}"
        mkdir -p "$DESTINATION/third-party-licenses/$(dirname "$relative")"
        cp -pL "$notice" "$DESTINATION/third-party-licenses/$relative"
    done < <(find "$SOURCE/$dependency_tree" -name .git -prune -o \
        \( -type f -o -type l \) \
        \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \
           -o -iname 'COPYRIGHT*' \) -print0)
done

# TTNN's optional golden comparisons and tracing import these support modules.
for support in models/__init__.py models/common/__init__.py \
    models/common/utility_functions.py models/common/tensor_utils.py tools/tracy; do
    if [ -e "$SOURCE/$support" ]; then
        mkdir -p "$DESTINATION/$(dirname "$support")"
        (cd "$SOURCE" && tar --exclude=__pycache__ -cf - "$support") |
            (cd "$DESTINATION" && tar -xf -)
    fi
done

while IFS= read -r -d '' link; do
    if [ ! -e "$link" ]; then
        echo "Packaged runtime contains a dangling symlink: $link" >&2
        exit 1
    fi
    case "$(realpath "$link")" in
        "$DESTINATION"/*) ;;
        *)
            echo "Packaged runtime symlink escapes its directory: $link" >&2
            exit 1
            ;;
    esac
done < <(find "$DESTINATION" -type l -print0)
