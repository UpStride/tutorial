# build docker image based on `dockerfile` and tag it as `upstride/tutorials:1.0`
build:
	docker build -t upstride/tutorials:1.0 .

# create a new docker container from docker image `upstride/tutorials:1.0` and attach the terminal session to it
# arguments to docker run:
# -it : attach an interactive session to the docker container
# --gpus all : enable all locally available GPUs in the docker, remove if the system has no GPUs
# --privileged : grants extra root capabilities, useful e.g. for TensorBoard profiling
# -v <local_path>:<docker_path> : mounts a local path in the docker container, can be used multiple times
# --rm : not used here, removes the docker container after it is exited from, otherwise docker containers stack up
run:
	@docker run -it --gpus all --privileged \
		-v $$(pwd):/opt \
        upstride/tutorials:1.0

        #  add the line below before the docker image to use your own dataset
        # -v /local_path_to_your_dataset/:/docker_path_to_your_dataset