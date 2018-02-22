#!/bin/bash

#-------------------------------------------------------------------------------
# update paths as needed for your local configuration
#-------------------------------------------------------------------------------
#DATA_DIR='/home/neilf/Fendley/adversarial/ISC_AML_2018/image_sets/val_prepped'
DATA_DIR=$1
OUTPUT_DIR='./submission_run'

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
# run attack! (if we haven't already)
#-------------------------------------------------------------------------------
if [ ! -d $OUTPUT_DIR ]; then
    PYTHONPATH=./cleverhans python sample_attack.py $DATA_DIR $OUTPUT_DIR 1 5
fi

# zip up the submission
cd $OUTPUT_DIR && zip -r ../sample_attack_fgm.zip ./*
