#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
# Treat unset variables as an error and exit immediately
set -u
# Return the exit status of the last command in the pipeline that failed, or zero if no command failed
set -o pipefail

error_handler() { 
  echo "Oops, looks like the script couldn't complete execution. I suggest following the manual install procedure available on 
  readme.md"; exit 1 
}

trap error_handler ERR

# Clone the GitHub repository
git clone https://github.com/NihalNavath/YOLOV8-train-on-detrac-dataset

# Download the data and annotations files using curl
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip
curl -O https://detrac-db.rit.albany.edu/Data/DETRAC-Test-Annotations-XML.zip

# Unzip the files using unzip with quiet option
unzip -q DETRAC-train-data.zip
unzip -q DETRAC-test-data.zip
unzip -q DETRAC-Train-Annotations-XML.zip
unzip -q DETRAC-Test-Annotations-XML.zip

# Remove the zip files
rm -rf DETRAC-test-data.zip
rm -rf DETRAC-train-data.zip
rm -rf DETRAC-Train-Annotations-XML.zip
rm -rf DETRAC-Test-Annotations-XML.zip

# Change directory to the cloned repository
cd YOLOV8-train-on-detrac-dataset
# Pull the latest changes from the remote repository
git pull
# Change directory to the preprocessor folder
cd preprocessor
# Run the parser.py script using python3
python3 parser.py

# Change directory to the cloned repository
cd ../
# Download the yolov8m.pt file using curl
curl -O https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Install the ultralytics package using pip
pip3 install ultralytics

echo "Wowie! the setup script has run successfully! you can now run python train.py -p PATH_TO_PT_FILE from root dir of cloned project."
