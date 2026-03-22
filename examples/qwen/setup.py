# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Download and prepare Qwen 2.5 0.5B-Instruct weights for tt-lang.

This script downloads the model from HuggingFace, extracts weights,
pads to tile boundaries, pre-transposes for tt-lang matmul, and saves
a checkpoint + tokenizer locally.

Usage:
    source build/env/activate
    python examples/qwen/setup.py
"""

import os
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "qwen2.5-0.5b.pt")
TOKENIZER_PATH = os.path.join(WEIGHTS_DIR, "tokenizer")


def is_ready():
    """Check if weights and tokenizer are already downloaded."""
    return (
        os.path.exists(CHECKPOINT_PATH)
        and os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer.json"))
    )


def setup():
    if is_ready():
        size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        print(f"Weights already present ({size_mb:.0f} MB). Nothing to do.")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"  Tokenizer:  {TOKENIZER_PATH}")
        return

    # Import here so the check above works without transformers installed
    try:
        from weight_extractor import extract_weights
    except ImportError:
        # Running from repo root
        sys.path.insert(0, os.path.dirname(__file__))
        from weight_extractor import extract_weights

    print("=" * 60)
    print("  Qwen 2.5 0.5B-Instruct — Setup")
    print("=" * 60)
    print()
    print("This will download ~1 GB from HuggingFace and prepare")
    print("the weights for tt-lang execution.")
    print()

    extract_weights(CHECKPOINT_PATH)

    print()
    print("=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Tokenizer:  {TOKENIZER_PATH}")
    print()
    print("Run the demo:")
    print("  python examples/qwen/chat_demo.py 2>/dev/null")


if __name__ == "__main__":
    setup()
