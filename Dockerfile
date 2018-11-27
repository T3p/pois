# python 3 + tensorflow 1
FROM tensorflow/tensorflow:latest-py3

# Required packages
RUN apt-get -y update && apt-get -y install ffmpeg git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv swig

# Working directory setup
ENV CODE_DIR /root/code
COPY . $CODE_DIR/pois

# Install baselines
RUN cd $CODE_DIR/pois && \
    rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

#Install rllab
ADD https://github.com/rll/rllab/archive/master.zip $CODE_DIR
RUN cd $CODE_DIR && \
    unzip master.zip && \
    cd rllab-master && \
    rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

# Set python 3 as default
RUN echo "alias python=python3" >> /root/.bashrc

# Set working directory
WORKDIR $CODE_DIR/pois/baselines

# Prepare interactive shell
CMD /bin/bash

