build:
    docker build -t upstride/tutorials:1.0  -f dockerfile .

run:
    @docker run -it --rm --gpus all --privileged \
        -v $$(pwd):/opt \
        -v /path_to_upstride_python_engine_repo/:/upstride_python \ # if users wishes to add custom layer. requires a pip install or add upstride folder to python path
        -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
        upstride/tutorials:1.0 \
        bash

# run_dcn:
#     @docker run -it --rm --gpus all --privileged \
#         -v $$(pwd):/opt \
#         -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
#         upstride/tutorials:1.0 \
#         bash

# run_transformers:
#     @docker run -it --rm --gpus all --privileged \
#         -v $$(pwd):/opt \
#         -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
#         upstride/tutorials:1.0 \
#         bash