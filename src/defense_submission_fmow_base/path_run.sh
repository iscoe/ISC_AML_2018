#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#
INPUT_DIR="/home/neilf/Fendley/adversarial/ISC_AML_2018/test_images"
OUTPUT_FILE="/home/neilf/Fendley/adversarial/ISC_AML_2018/output/preds.csv"
./run_defense.sh ${INPUT_DIR} ${OUTPUT_FILE}