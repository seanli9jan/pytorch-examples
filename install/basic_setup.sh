#!/bin/bash

sudo apt install -y --no-install-recommends openssh-server \
                                            build-essential \
                                            python3-dev \
                                            python3-pip \
                                            cifs-utils  \
                                            shc

cd /usr/bin \
    && sudo ln -s python3 python \
    && sudo ln -s pip3 pip

pip --no-cache-dir install Cython
