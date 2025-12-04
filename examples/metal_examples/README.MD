# To Run Metal Examples
Examples are meant to be run on a machine with a single n150 card. Set the env value TT_VISIBLE_DEVICES to be the pcie card you want to use in a multi-device setting.
clone and build [metal](https://github.com/tenstorrent/tt-metal).
build metal env and source it using the metal repos provided script (for ttnn module).
```bash
./create_env.sh
source ./python_env/bin/activate
```
Now go to the tt-lang and run the desired metal kernels, such as the singlecore matmul kernel.
```bash
pytest ./examples/metal_matmul.py::test_singlecore_matmul_metal
```

# TT-Lang Examples
any tt-lang in this folder is up to spec, but currently is not guaranteed to compile/execute
