#!/bin/bash

. env.sh

CONTAINER_WORKDIR=/home/user/drl
PYTHON_CMD="python navigation.py run"

cmd="conda activate $CONDA_ENV && \
cd $CONTAINER_WORKDIR && \
$PYTHON_CMD"

echo $cmd

docker run -it -v $(pwd):$CONTAINER_WORKDIR $DOCKER_IMAGE_TAG bash -c $cmd