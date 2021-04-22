build:
	docker build -t upstride/tutorials:1.0 .

run:
	@docker run -it --rm --gpus all --privileged \
		-v $$(pwd):/opt \
        -v /path_to_upstride_python_engine_repo/:/upstride_python \
        -v ~/path_to_your_dataset/:/path_to_your_dataset \
        upstride/tutorials:1.0 \
		bash