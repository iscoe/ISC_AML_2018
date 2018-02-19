#!/bin/bash
DATA_DIR='/home/fendlnm1/Fendley/adversarial/ISC_AML_2018/image_sets/val_prepped'
OUTPUT_DIR='./adv_out'
EPS=[1,2,3,4,5]
python sample_attack.py $DATA_DIR $OUTPUT_DIR $EPS