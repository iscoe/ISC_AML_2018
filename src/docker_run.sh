#!/bin/bash
INPUT_IMAGES="/home/neilf/Fendley/adversarial/attackFmow/test_images"
OUTPUT_DATA="/home/neilf/Fendley/adversarial/attackFmow/output"
sudo nvidia-docker run \
	-v ${INPUT_IMAGES}:/input_images \
	-v ${OUTPUT_DATA}:/output \
	-e USERID=$UID simple_defense \
	./run_defense.sh \
	/input_images \
	/output
