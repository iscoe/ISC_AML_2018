#!/bin/bash
if [ ! -f cnn_image_only.model ]; then
    wget https://github.com/fMoW/baseline/releases/download/paper/cnn_image_only.model.zip -O cnn_image_only.model.zip
    unzip cnn_image_only.model.zip
    rm -f cnn_image_only.model.zip
fi

tar cvzf simple_defense_submission.tgz cnn_image_only.model data_ml_functions/ metadata.json run_defense.sh defense_zip.sh process.py
