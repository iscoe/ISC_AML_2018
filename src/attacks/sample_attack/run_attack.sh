#!/bin/bash

#-------------------------------------------------------------------------------
# update paths as needed for your local configuration
#-------------------------------------------------------------------------------
DATA_DIR=$1
OUTPUT_DIR='./submission_run'
ATTACK=$2

#-------------------------------------------------------------------------------
# dependencies
#-------------------------------------------------------------------------------

# check out a local copy of cleverhans (CH), if one does not already exist.
# if your python already has CH installed, you may want to omit this.
if [ ! -d './cleverhans' ]; then
    echo "checkout CH..."
    git clone https://github.com/tensorflow/cleverhans.git
fi


# weights for the model we are attacking
if [ ! -f cnn_image_only.model ]; then
    wget https://github.com/fMoW/baseline/releases/download/paper/cnn_image_only.model.zip -O cnn_image_only.model.zip
    unzip cnn_image_only.model.zip
    rm -f cnn_image_only.model.zip
fi


#-------------------------------------------------------------------------------
# run attack 
#-------------------------------------------------------------------------------

# first, delete any old output from previous runs.
if [ -d $OUTPUT_DIR ]; then
    \rm -rf $OUTPUT_DIR
fi

PYTHONPATH=./cleverhans python sample_attack.py $DATA_DIR $OUTPUT_DIR $ATTACK 0 1 3

# zip up the submission
cd $OUTPUT_DIR && zip -r ../sample_attack_fgm.zip ./*
