#!/bin/bash

sudo nvidia-docker run --privileged            \
-it --hostname docker --name seanli9jan        \
-e COLUMNS=$(tput cols) -e LINES=$(tput lines) \
-v $HOME/docker/workdir:/root/workdir          \
-p 6006:6006 -p 8888:8888                      \
seanli9jan/pytorch:cuda10.1-cudnn7-devel-ubuntu18.04 bash
