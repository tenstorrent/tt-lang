# To Run Metal Examples
Examples are meant to be run on a machine with a single n150 card. Set the env value TT_VISIBLE_DEVICES to be the pcie card you want to use in a multi-device setting.
Manual build of tt-mlir needed, to source ttnn module
in tt-mlir, source env/activate
Now go to tt-lang, source build/env/activate and run the desired metal kernels, such as the singlecore matmul kernel.
```bash
pytest ./examples/metal_examples/singlecore_matmul/metal/singlecore_matmul.py
```

# TT-Lang Examples
any tt-lang in this folder is up to spec, but currently is not guaranteed to compile/execute
