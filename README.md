# Custom Robosuite Environments

Custom robosuite environments for data collection


## Introduction

You can easily collect datasets by using the below command
```zsh
$ uv run mjpython scripts/collect_demonstrations.py --directory data/ --environment StackThreeCubes --device spacemouse
```

You can also check the collected datasets by using the following command
```zsh
$ uv run python scripts/playback_demonstrations.py --folder data/{file_name}
```