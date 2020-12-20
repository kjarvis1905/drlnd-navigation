#!/bin/bash

FILE=`readlink -f $0`
DIR=`dirname $FILE`
ENV_NAME="drlndtest"

# Setup the environment

ENV_FILE="$DIR/conda_env.yml"

if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit
fi

echo "Creating conda environment"

conda create --name $ENV_NAME python=3.6

if [ $? -eq 0 ]
then
    echo "Success!"
fi

conda activate $ENV_NAME

DRLND_URL="https://github.com/udacity/deep-reinforcement-learning/"

echo "Installing requirements from $DRLND_URL"

python -m pip install -e "git+$DRLND_URL#egg=unityagents&subdirectory=python"

conda install --name $ENV_NAME --file "conda_requirements.txt" 

