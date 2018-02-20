#!/bin/bash

DATA_DIR='/home/neilf/Fendley/adversarial/ISC_AML_2018/image_sets/FMOW_1000_CORRECT'
OUTPUT_DIR='./submission_run'

#-------------------------------------------------------------------------------
# dependencies
#-------------------------------------------------------------------------------

# check out a local copy of cleverhans (CH), if one does not already exist.
# if your python already has CH installed, you may want to omit this.
if [ ! -f ./cleverhans ]; then
    git clone https://github.com/tensorflow/cleverhans.git
fi

# weights for the model we are attacking
if [ ! -f cnn_image_only.model ]; then
    wget https://github.com/fMoW/baseline/releases/download/paper/cnn_image_only.model.zip -O cnn_image_only.model.zip
    unzip cnn_image_only.model.zip
    rm -f cnn_image_only.model.zip
fi

#-------------------------------------------------------------------------------
# run attack!
#-------------------------------------------------------------------------------
PYTHONPATH=./cleverhans python sample_attack.py $DATA_DIR $OUTPUT_DIR 0 1 2 3 4 5 10 15 20 
