#!/bin/bash
INPUT_IMAGES=/home/neilf/Fendley/adversarial/ISC_AML_2018/test_images
OUTPUT_DATA=/home/neilf/Fendley/adversarial/ISC_AML_2018/output/
DOCKER_CONTAINER=simpledefense
sudo nvidia-docker run \
	-v ${INPUT_IMAGES}:/input_images \
	-v ${OUTPUT_DATA}:/output \
	--user www-data $DOCKER_CONTAINER ./run_defense.sh \
	/input_images \
	/output/predictions.npy
