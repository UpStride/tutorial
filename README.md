# Tutorials

This repository contains three tutorials that use the upstride engine.

- *Simple network*: this is a basic example on how to train a simple convolutional neural network (CNN) using the upstride engine. We also provide guidance on how to easily convert your own neural networks (NNs) from real to hyper-complex using the upstride layers.
- *Deep complex networks*: this tutorial shows how to reproduce the seminal paper of Trabelsi et al., called "Deep Complex Networks" (DCN), using the upstride engine and [upstride's classification API](https://github.com/UpStride/classification-api).
- *Quaternion transformers*: this tutorial provides a simple codebase that showcases how to implement and use a quaternion-valued transformer networks for the popular tasks of sentiment analysis and neural machine translation.

## System Requirements

- System: Linux (preferably Ubuntu)
- Python version: 3.6 or later
- Nvidia GPU preferable for fast training

If docker is to be used:

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA-container-runtime](https://nvidia.github.io/nvidia-container-runtime/), with a [useful source on installation steps](https://github.com/NVIDIA/nvidia-docker/issues/1243#issuecomment-694981577)

Otherwise:

- local Nvidia libraries compatible with TensorFlow 2.4 if Nvidia GPU is used

## Installation

Before diving into the examples, let's set up the environment!

If you wish to install the required packages directly on you machine, follow *method 1*. If instead you wish to work in a docker container, use *method 2*.

Local installation should hopefully go well, but it is possible that some problems with e.g. Nvidia libraries might emerge - the engine is meant to work with TensorFlow 2.4, which has specific Nvidia dependencies. That is why we also provide the docker approach, which should work regardless of the local configuration and libraries installed in the system.

### Method 1 (local)

> pip install -r requirements.txt

The packages will be installed locally.

### Method 2 (docker)

Use dockerfile to build and run the image without installing the dependencies on the local system.

#### Build and run

Let's prepare the environment by building the docker. We use the `makefile` to build docker images and run docker containers.

From the root of this directory type in the shell:

```bash
make build
```

You would see the initial docker image being pulled from registry and dependencies being installed to construct a new docker image.

To run a new docker container based on the built image, use the command below:

```bash
make run
```

A docker container will be launched and you should see the docker terminal in the same window. Note: You will be logged in as root.

#### Docker environment introduction

This section serves as an introduction for working with dockers, it is helpful especially when some changes to the dockers' configuration are necessary.

`dockerfile` is a file that contains the configuration of the docker, with the required dependencies to be installed. We include the same packages as in `requirements.txt`, which are necessary to run the simple-network and quaternion-transofmers examples. Note: there are some further requirements for the deep-complex-networks example, handled in the classification-api repository.

In the `build` command we construct the docker image from `dockerfile` and tag it as `upstride/tutorials:1.0`.

Here is the `run` command from the makefile:

```make
run:
    @docker run -it --rm --gpus all \
    -v $$(pwd):/opt \
    upstride/tutorials:1.0
```

In the `run` command we use the following arguments:

+ `-it` : attach an interactive session to the docker container
+ `--gpus all` : enable all locally available GPUs in the docker, remove this flag if the system has no GPUs
+ `--privileged` : grants extra root capabilities, useful e.g. for TensorBoard profiling
+ `-v <local_path>:<docker_path>` : mounts a local path in the docker container, in our case we mount the directory under the current path (`pwd`) as the `/opt` directory inside the docker container. This flag can be used multiple times, e.g. it can be used to mount datasets or other repositories.

Other useful flags:

+ `--rm` : it removes the docker container after it is exited from, otherwise docker containers are not destroyed and still take drive memory
+ `--name <name>` : docker container will be named accordingly

After a docker container that was invoked without the `--rm` flag is turned off, it can be revived using:

> docker start <docker container id / name>

> docker attach <docker container id / name>

Running dockers (with their ids and names) can be checked using:

> docker ps

To check all the docker containers in the system run:

> docker ps -a

Further terminal sessions can be attached to a running docker container using:

> docker exec -it <docker container id / name> bash

Refer to the [docker documentation](https://docs.docker.com/engine/reference/run/) for further information.

## Upstride engine

To run the tutorials, it's necessary to clone the upstride engine.

If dockers are to be used, we recommend cloning the engine into this tutorials directory - this way the engine code will be available in the docker container with no alterations to the `makefile`. Alternatively, the engine code located under a different local path can be mounted in the docker using the `-v` argument in the `makefile`. Note: for the deep-complex-network example, the same applies to  classification-api code: it needs to either be cloned into this directory or mounted using the `-v` argument.

To use the engine `PYTHONPATH=<local_path_to_upstride_engine>` needs to be appended before running the training scripts, e.g. `PYTHONPATH=/opt/upstride_engine python train.py`. This is required so that you can correctly import upstride modules.

Note: You can also set the `PYTHONPATH` to the <local_path_to_upstride_engine> by typing the line below in terminal:
```bash
export PYTHONPATH=<local_path_to_upstride_engine>
```
You only need to do this once per terminal session.

Alternatively, you could perform `pip install -e .` from the root of the upstride_python directory which would install the upstride engine in the users' python environment. Upstride modules will then be automatically picked up without having to specify `PYTHONPATH`.

## Usage

To get started with the chosen tutorial, please refer to the instructions in the corresponding folder.
