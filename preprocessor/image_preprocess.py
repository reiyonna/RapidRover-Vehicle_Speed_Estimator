import os
import shutil

from os import listdir

import sys
sys.path.append(os.path.dirname("../"))
from utils.path import path_exists, remove_if_exists, make_dirs

train_data_path = "../../Insight-MVT_Annotation_Train/" 
test_data_path = "../../Insight-MVT_Annotation_Test/" 

if not os.path.exists(train_data_path):
    train_data_path = "../../DETRAC-train-data/Insight-MVT_Annotation_Train/" 
    test_data_path = "../../DETRAC-test-data/Insight-MVT_Annotation_Test/" 

def run_image_preprocess():
    check_if_ready()

    # start preprocessing training images
    print("Running image processing for Train dataset... this could take some time.")
    for folder in listdir(train_data_path):
        for image in listdir(train_data_path + f"/{folder}"):
            shutil.copy(train_data_path + f"/{folder}" + f"/{image}", f"../output/images/train/{image[:-4]}{folder}.jpg")
    
    print("TRAINING IMAGES PREPROCESSED!")

    # start preprocessing evaluation images
    print("Running image processing for Test dataset... this could take some time.")
    for folder in listdir(test_data_path):
        for image in listdir(test_data_path + f"/{folder}"):
            shutil.copy(test_data_path+ f"/{folder}" + f"/{image}", f"../output/images/val/{image[:-4]}{folder}.jpg")

    print("--SUCCESS! IMAGE PREPROCESS DONE!--")

    
def check_if_ready():
    if not path_exists('../temp'):
        raise Exception("XML preprocessor has not run yet")
    
    # check if program can find train and test datasets
    for data_path, data_type in [(train_data_path, "training"), (test_data_path, "testing")]:
        if not os.path.exists(data_path):
            raise Exception(f"The script cannot see the {data_type} dataset in the provided directory!")
     
    if path_exists("../output"):
        remove_if_exists("../output", message="Found existing output folder, removing...")

    # create output folder with necessary subfolder
    folders = [
        "../output/images/val",
        "../output/images/train",
        "../output/labels/val",
        "../output/labels/train"
    ]

    make_dirs(folders)