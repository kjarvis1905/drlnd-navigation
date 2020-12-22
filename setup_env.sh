#!/bin/bash

FILE=`readlink -f $0`
DIR=`dirname $FILE`
ENV_NAME="drlndtest"

# Eval the conda bash hook

eval "$(conda shell.bash hook)"

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

eval "pip install gym['box2d']"

DRLND_URL="https://github.com/udacity/deep-reinforcement-learning/"

echo "Installing requirements from $DRLND_URL"

pip install -e "git+$DRLND_URL#egg=unityagents&subdirectory=python"

pip install -r "$DIR/conf/extra_requirements.txt"

