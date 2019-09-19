#!/bin/bash

sudo docker build -t seanli9jan/pytorch:cuda10.1-cudnn7-devel-ubuntu18.04 . -f Dockerfile.devel-gpu
