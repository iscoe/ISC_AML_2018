#!/bin/bash

DATA_DIR='/home/neilf/Fendley/adversarial/ISC_AML_2018/image_sets/FMOW_1000_CORRECT'
OUTPUT_DIR='./submission_run'
python sample_attack.py $DATA_DIR $OUTPUT_DIR 0 1 2 3 4 5 10 15 20 