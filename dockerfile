FROM eu.gcr.io/fluid-door-230710/upstride:py-1.1.1-tf2.3.0-gpu 
# TODO replace the above with the updated python engine

RUN apt-get update && \
    pip install \
    tensorflow-gpu==2.4.1 \
    tensorflow-datasets==4.2.0 \
    tensorflow-text==2.4.1 \
    tabulate==0.8.9 \
    nltk==3.6.1 \
    pyyaml \
    upstride_argparse \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /opt