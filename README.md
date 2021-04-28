# Tutorials

This repository contains three tutorials that use the upstride engine.

- *Simple network*: this is a basic example on how to train a simple convolutional neural network (CNN) using the upstride engine. We also provide guidance on how to easily convert your own neural networks (NNs) from real to hyper-complex using the upstride layers.
- *Deep complex networks*: this tutorial shows how to reproduce the seminal paper of Trabelsi et al., called "Deep Complex Networks" (DCN), using the upstride engine and [upstride's classificatio API](https://github.com/UpStride/classification-api).
- *Quaternion transformers*: this tutorial provides a simple codebase that showcases how to implement and use a quaternion-valued transformer networks for the popular tasks of sentiment analysis and neural machine translation.

## System Requirements

- System: Linux (preferably Ubuntu)
- Python version: 3.6 or later
- [Docker](https://docs.docker.com/engine/install/)
- Nvidia GPU is preferable for fast training
- [NVIDIA-container-runtime](https://nvidia.github.io/nvidia-container-runtime/)

## Installation

Before diving into the examples, let's set up the environment!

If you wish to install the required packages directly on you machine, follow *method 1*. If instead you wish to work in a docker container, use *method 2*.

Local installation should hopefully go well, but it is possible that some problems with e.g. Nvidia libraries will emerge - the engine uses TensorFlow 2.4, which has specific Nvidia dependencies. That is why we also provide the docker approach, which should work regardless of the local configuration and libraries installed in the system.

### Method 1 (local)

> pip install -r requirements.txt

The packages will be installed locally.

### Method 2 (docker)

Use dockerfile to build and run the image without installing the dependencies on the local system.

#### Build the docker file

Let's prepare the environment by building the docker. We use the `makefile` to build and run the dockers

From the root of this directory type in the shell:
> make build

You would see the initial docker image being pulled from registry and dependencies being installed to construct a new docker image.

#### Launch the docker

From the shell run the below command:
> make run

The docker would be launched and you should see the docker terminal in the same window. Note: You will be logged in as root.

#### Docker environment introduction

This section serves as an introduction to docker manipulation, especially if some changes have to be introduced.

`dockerfile` is the file that contains the configuration of the docker, with the required dependencies to install. We include the same packages as in `requirements.txt`, which are necessary to run the simple-network and quaternion-transofmers examples (there are some further requirements for the deep-complex-networks example, handled in the classification-api repository)

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
+ `-v <local_path>:<docker_path>` : mounts a local path in the docker container, this flag can be used multiple times. It can be used e.g. to mount datasets or other repositories.

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

Refer to the [docker documentation](https://docs.docker.com/engine/reference/commandline/run/#options) for further information.

## Upstride engine

To run the tutorials, it's necessary to clone the upstride engine.

If dockers are to be used, we recommend cloning the engine into this tutorials directory - this way the makefile will not have to be altered in order to include the engine code in the docker container. Alternatively, the engine code located under a different local path can be mounted in the docker using the `-v` argument in the `makefile`. Note: for the deep-complex-network example, the same applies to  classification-api code: it needs to either be cloned into this directory or mounted using the `-v` argument.

To use the engine `PYTHONPATH=<local_path_to_upstride_engine>` needs to be appended before running the training scripts, e.g. `PYTHONPATH=/opt/upstride_engine python train.py`. This is required so that you can correctly import upstride modules.

Note: You can also set the `PYTHONPATH` to the <local_path_to_upstride_engine> by typing the below in terminal:
```bash
export PYTHONPATH=<local_path_to_upstride_engine>
```
you only need to do this once per terminal session.

Alternatively, you could perform `pip install -e .` from the root of the upstride_python directory which would install the python engine in the users' python environment.

## Usage

To get started with the chosen tutorial, please refer to the instructions in the corresponding folder.
