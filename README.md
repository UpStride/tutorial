# Tutorials

This repository contains examples and step by step guide on:
* How to use UpStride Engine
* Examples on how to adapt the code to your project

## Installation 

There are 2 ways to setup the environment. 

__Method 1__

> pip install -r requirements.txt


If you have access to the upstride_python repository then you would need to append the `PYTHONPATH="local_path_where_the_upstride_python_repo" python train.py` before invoking the python script. This is required so that you can import upstride modules without import errors. 

Only Python 3.6 or later is supported.

__Method 2__

use dockerfile to build and run the image without installing the dependencies on the local system. 

__Build the docker file__

Let's prepare the environment by building the docker file. 
The `dockerifle` contains the UpStride python engine and required dependencies to be installed.

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

For basic example: [README.md](./basic/README.md)
For DCN(Deep Complex Networks) example: [README.md](./deep-complex-networks/README.md)
For Quaternion Transformers example: [README.md](./quaternion-transformers/README.md)
