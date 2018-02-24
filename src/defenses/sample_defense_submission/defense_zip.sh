#!/bin/bash
#
# Creates a defense submission (.zip file).
# Note the metadata.json is needed so we know how to run the code in Docker!


if [ ! -f cnn_image_only.model ]; then
    wget https://github.com/fMoW/baseline/releases/download/paper/cnn_image_only.model.zip -O cnn_image_only.model.zip
    unzip cnn_image_only.model.zip
    rm -f cnn_image_only.model.zip
fi

zip defense_noop_submission.zip cnn_image_only.model metadata.json run_defense.sh defense_zip.sh process.py
