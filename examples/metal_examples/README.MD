# To Run Metal Examples
Examples are meant to be run on a machine with a single n150 card. Set the env value TT_VISIBLE_DEVICES to be the pcie card you want to use in a multi-device setting.
Build tt-lang with hardware support, source `build/env/activate`, and run the
desired metal kernels, such as the singlenode matmul kernel.
```bash
pytest ./examples/metal_examples/singlenode_matmul/metal/singlenode_matmul.py
```

# TT-Lang Examples
any tt-lang in this folder is up to spec, but currently is not guaranteed to compile/execute
