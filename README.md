# Learning Sturctured Commnunication

A Tensorflow implementation of `LSC`.

 
## Code structure

- `./graph_nets`: contains code for establishing communication sturcture.

- `./examples/`: contains scenarios for Ising Model and Battle Game (also models).

- `train_battle.py`: contains code for training Battle Game models

## Requirements Installation
```shell
pip install ./
```
## Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent)

**Steps for compiling**

```shell
cd examples/battle_model
./build.sh
```

**Steps for training models under Battle Game settings**

1. Add python path in your `~/.bashrc` or `~/.zshrc`:

    ```shell
    vim ~/.zshrc
    export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
    source ~/.zshrc
    ```

2. Run training script for training:

    ```shell
    ./runtiny.sh
    ```
