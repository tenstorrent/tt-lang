# To Run
clone metal, or checkout to arichins/splitWorkUtilPybind
build the repo
build metal env
source metal env (for ttnn module)
run pytest in this repo
```bash
git clone -b arichins/splitWorkUtilPybind https://github.com/tenstorrent/tt-metal.git
    or
git checkout arichins/splitWorkUtilPybind
cd ./tt-metal
./build-metal.sh
./create_env.sh
source ./python_env/bine/activate
cd ../tt-lang
    run whatever pytest in examples/metal_examples
```