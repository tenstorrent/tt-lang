# To Run
clone metal
build the repo
build metal env
source metal env (for ttnn module)
run pytest in this repo
```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd ./tt-metal
./build-metal.sh
./create_env.sh
source ./python_env/bin/activate
cd ../tt-lang
    run whatever metal pytest in examples
```

any tt-lang in this folder is up to spec, but currently is not guaranteed to compile/execute
