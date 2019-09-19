#!/bin/bash

###
## Install docker
# Uninstall old versions
sudo apt-get remove docker docker-engine docker.io

# Set up the repository
sudo apt update

sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common

#curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     xenial \
     stable"

# Install Docker CE
sudo apt update
sudo apt install docker-ce

#apt-cache madison docker-ce
#sudo apt-get install docker-ce=<VERSION>

# Test docker
#sudo docker run hello-world

# Uninstall Docker CE
#sudo apt-get purge docker-ce
#sudo rm -rf /var/lib/docker

###
## Install nvidia-docker (old)
# Need to install nvidia drivers and docker before install nvidia-docker

#sudo apt install nvidia-modprobe

# Install nvidia-docker and nvidia-docker-plugin
#wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
#sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Removing nvidia-docker 1.0
#docker volume ls -q -f driver=nvidia-docker | \
#xargs -r -I{} -n1 docker ps -q -a -f volume={} | \
#xargs -r docker rm -f
#sudo apt purge nvidia-docker

## Install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

sudo apt install nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi
#sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

# Manage docker as a non-root user
#sudo groupadd docker
#sudo usermod -aG docker $USER

###
## After install docker or nvidia-docker
# Download images
#docker pull tensorflow/tensorflow
# Download latest gpu images
#docker pull tensorflow/tensorflow:latest-gpu

# Create container
#NV_GPU=0 nvidia-docker run -it --hostname docker --name $USER           \
                           #-v /home/$USER/docker/workdir:/root/workdir  \
                           #-w /root -p 6006:6006 -p 8888:8888           \
                           #tensorflow/tensorflow:latest-gpu bash

# Start container
#docker start $USER

# Run container
#docker exec -it $USER bash

# Stop container
#docker stop $USER

# Remove container
#docker rm $USER

###
## In continer
# Install jupyter themes
#pip install --upgrade jupyterthemes
#jt -l          # list available themes
#jt -t oceans16 # theme name to install
#jt -r          # reset to default theme

# Run jupyter
#source /root/.bashrc && export SHELL=/bin/bash && jupyter notebook --allow-root

# Run jupyter in the background
#export SHELL=/bin/bash
#nohup jupyter notebook --allow-root > /root/jupyter.log &

# Docker terminal style
#echo "### My setting ###" >> /root/.bashrc
#echo "PS1='[\[\e[31;1m\]\u\[\e[32;1m\]@\h \[\033[01;34m\]\W\[\e[0m\]]\[\e[31;1m\]$\[\e[0m\] '" >> /root/.bashrc

###
## Jupyter setting
# Jupyter passwd (python code)
#from notebook.auth import passwd
#print(passwd())

# Jupyter passwd
# Add "c.NotebookApp.password = u'sha1:12...'" to jupyter_notebook_config.py
# sha1:12... is created by print(passwd())

# Jupyter login path
# Add "c.NotebookApp.notebook_dir = u'/root/'" to jupyter_notebook_config.py

