#!/bin/bash
#
# Creates a defense submission (.zip file).
# Note the metadata.json is needed so we know how to run the code in Docker!


#--------------------------------------------------
# Download the CNN weights, if we haven't already.
#--------------------------------------------------
if [ ! -f cnn_image_only.model ]; then
    wget https://github.com/fMoW/baseline/releases/download/paper/cnn_image_only.model.zip -O cnn_image_only.model.zip
    unzip cnn_image_only.model.zip
    rm -f cnn_image_only.model.zip
fi


#--------------------------------------------------
# Create two different submissions (one for each baseline); 
# however, you'll only need one.
#--------------------------------------------------
echo "[info]: creating NOOP defense"
cp run_defense_noop.sh run_defense.sh
zip defense_noop_submission.zip cnn_image_only.model metadata.json run_defense.sh defense_zip.sh process.py

echo "[info]: creating simple averaging defense"
cp run_defense_avg.sh run_defense.sh
zip defense_avg_submission.zip cnn_image_only.model metadata.json run_defense.sh defense_zip.sh process.py

