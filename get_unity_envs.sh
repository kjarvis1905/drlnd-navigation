#!/bin/bash

FILE=`readlink -f $0`
DIR=`dirname $FILE`
UNITY_ENV_DIR="unity_environments"
UNITY_ENV_URL="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana"

if [ ! -d $UNITY_ENV_DIR ]
then
    mkdir $UNITY_ENV_DIR
fi

cd $UNITY_ENV_DIR

get_unity_env () {
echo "Looking for ./$UNITY_ENV_DIR/$1"
if [ ! -d "$DIR/$UNITY_ENV_DIR/$1" ]
then
    echo "Downloading and unzipping environment: $1"
    wget "$UNITY_ENV_URL/$1.zip"
    unzip "$1.zip"
else
    echo "$DIR/$UNITY_ENV_DIR/$1 exists, skipping"
fi
}

declare -a envs=("Banana_Linux" "Banana_Linux_NoVis" "VisualBanana_Linux")
for ENV in "${envs[@]}"
do
    get_unity_env $ENV
done
