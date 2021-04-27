build:
	docker build -t upstride/tutorials:1.0 .

run:
	@docker run -it --rm --gpus all --privileged \
		-v $$(pwd):/opt \
        upstride/tutorials:1.0 \
		bash
		
        #  add the below before the line containing 'bash' to use your own dataset
        # -v ~/path_to_your_dataset/:/path_to_your_dataset \