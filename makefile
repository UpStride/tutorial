build_upstride:
    docker build -t upstride/classification_api:upstride-1.0 -f dockerfiles/upstride.dockerfile .

build_dcn:
    docker build -t upstride/classification_api:dcn-1.0 -f dockerfiles/dcn.dockerfile .

build_transformers:
    docker build -t upstride/classification_api:transformers-1.0 -f dockerfiles/transformers.dockerfile .

run_upstride:
    @docker run -it --rm --gpus all --privileged \
        -v $$(pwd):/opt \
        -v /path_to_upstride_python_engine_repo/:/upstride_python \ # if users wishes to add custom layer. requires a pip install or add upstride folder to python path
        -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
        upstride/classification_api:upstride-1.0 \
        bash

run_dcn:
    @docker run -it --rm --gpus all --privileged \
        -v $$(pwd):/opt \
        -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
        upstride/classification_api:dcn-1.0 \
        bash

run_transformers:
    @docker run -it --rm --gpus all --privileged \
        -v $$(pwd):/opt \
        -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
        upstride/classification_api:transformers-1.0 \
        bash