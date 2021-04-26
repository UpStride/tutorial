# Tutorials

This repository contains three tutorials that use the upstride engine. 
* *Simple network*: this is a basic example on how to train a simple convolutional neural network (CNN) using the upstride engine. We also provide guidance on how to easily convert your own neural networks (NNs) from real to hyper-complex using the upstride layers.
* *Deep complex networks*: this tutorial shows how to reproduce the seminal paper of Trabelsi et al., called "Deep Complex Networks" (DCN), using the upstride engine and [upstride's classificatio API](https://github.com/UpStride/classification-api).
* *Quaternion transformers*: this tutorial provides a simple codebase that showcases how to implement and use a quaternion-valued transformer networks for the popular tasks of sentiment analysis and neural machine translation.

## Pre-requisites

All the tutorials in this repository require Python 3.6 or later.

## Installation 

Before diving into the examples, let's set up the environment! 

If you wish to install the required packages directly on you machine, follow *method 1*. If instead you wish to work in a docker container, use *method 2*.

__Method 1__ 

> pip install -r requirements.txt


If you have access to the upstride_python repository then you would need to append the `PYTHONPATH="local_path_where_the_upstride_python_repo" python train.py` before invoking the python script. This is required so that you can correctly import upstride modules. 

Alternatively, you could perform `pip install -e .` from the root of the upstride_python directory which would install the python engine in the users' python environment. This way any changes made to the upstride_python directory can be quickly tested or validated.

__Method 2__

use dockerfile to build and run the image without installing the dependencies on the local system. 

__Build the docker file__

Let's prepare the environment by building the docker file. 
The `dockerfile` contains the UpStride python engine and required dependencies to be installed.

We use the `makefile` to build and run the dockers

From the root of this directory type in the shell:
> make build

you would see the docker being pulled from our cloud registry, dependencies being installed and finally the docker image.

__Launch the docker__

From the shell run the below command:
> make run

The docker would be launched and you should see the docker terminal in the same window. Note: You will be logged in as root. 

If the user wishes to use a specific dataset, Ensure to mount them via `-v local_path:path_inside_the_docker` 


## Usage

To get started with the a tutorial, please refer to the instructions in the corresponding folder.
