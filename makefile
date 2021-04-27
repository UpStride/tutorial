build:
	docker build -t upstride/tutorials:1.0 .

run:
	@docker run -it --rm --privileged \
		-v $$(pwd):/opt \
        upstride/tutorials:1.0

        #  add the below before the docker image to use your own dataset
        # -v ~/path_to_your_dataset/:/path_to_your_dataset 