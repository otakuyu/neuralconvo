FROM ubuntu:14.04

RUN apt-get update && apt-get install -y \
    git \
    python-dev

WORKDIR ${HOME}
RUN git clone https://github.com/torch/distro.git ./torch --recursive

WORKDIR ${HOME}/torch

RUN /bin/bash install-deps;
RUN ./install.sh

RUN luarocks install nn
RUN luarocks install rnn
RUN luarocks install penlight