# No GPU:
FROM tensorflow/tensorflow:latest-py3

# Required packages
RUN apt-get -y update && apt-get -y install ffmpeg git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

# Working directory setup
ENV CODE_DIR /root/code
COPY . $CODE_DIR/baselines
WORKDIR $CODE_DIR/baselines

# Install baselines
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

# Set python 3 as default
RUN echo "alias python=python3" >> /root/.bashrc

WORKDIR $CODE_DIR/baselines/baselines

CMD /bin/bash

