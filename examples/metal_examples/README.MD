# To Run Metal Examples
Examples are meant to be run on a machine with a single n150 card. Set the env value `TT_VISIBLE_DEVICES` to be the pcie card you want to use in a multi-device setting, e.g., for card ID 2:
```bash
export TT_VISIBLE_DEVICES="2"
export TT_METAL_CACHE="$HOME/.cache/tt_metal_$TT_VISIBLE_DEVICES"
```

Now go to tt-lang, `source build/env/activate` and run the desired metal kernels, such as the singlecore matmul kernel.
```bash
pytest ./examples/metal_examples/singlecore_matmul/metal/singlecore_matmul.py
```

# TT-Lang Examples
Any tt-lang in this folder is up to spec, but currently is not guaranteed to compile/execute.
