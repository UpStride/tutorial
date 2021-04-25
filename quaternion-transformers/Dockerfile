FROM tensorflow/tensorflow:2.4.1-gpu
MAINTAINER Mohammad Reza Loghmani (mrloghmani@upstride.io)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y --upgrade vim
WORKDIR /transformers
COPY . .
RUN pip install --upgrade pip
RUN pip install tensorflow_datasets==4.2.0 tensorflow_text==2.4.3 tabulate==0.8.9 nltk==3.6.1