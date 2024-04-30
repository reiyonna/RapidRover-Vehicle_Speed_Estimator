from os import listdir, mkdir, path, getcwd
from shutil import rmtree
import shutil
from bs4 import BeautifulSoup
import glob

import os
import sys
sys.path.append(os.path.dirname("../"))
from utils.path import path_exists, remove_if_exists, make_dirs

import image_preprocess

import argparse
parser = argparse.ArgumentParser()

root_output_dir = "../output"
xml_loc_train =  "../../DETRAC-Train-Annotations-XML"
xml_loc_test =  "../../DETRAC-Test-Annotations-XML"
temp_annotations_loc =  "../temp"

if len(listdir(xml_loc_train)) < 5:
    xml_loc_train =  "../../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML"
    xml_loc_test =  "../../DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML"

assert len(listdir(xml_loc_train)) > 5
assert len(listdir(xml_loc_test)) > 5

train_backgrounds, test_backgrounds = 0, 0

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

def xml_to_yolov8_format(loc: str):
    """converts xml labels provided by DETRAC into .txt files as required by YOLO

    Args:
        loc (str): indicator for train/test dataset
    """
    location = xml_loc_train if loc == "train" else xml_loc_test

    global train_backgrounds, test_backgrounds

    for file_name in listdir(location):
        # use os.path.join to create file paths
        file_path = path.join(location, file_name)
        # use with open to read files
        with open(file_path, "r") as f:
            data = f.read()

        bs_dat = BeautifulSoup(data, "xml")

        print(f"preprocessing {file_name}")

        for index, frame in enumerate(bs_dat.find_all("frame")):
            number = int(frame["num"])
            for target in frame.find_all("target"):
                #id = target['id']

                # keep track of backgrounds
                if number != index + 1 and loc == "train":
                    train_backgrounds += 1
                elif number != index + 1 and loc == "test":
                    test_backgrounds += 1

                vehicle_type = target.find('attribute')['vehicle_type']

                # use dict.get to get the class index from the vehicle type
                class_index = {"car": 0, "van": 1, "bus": 2}.get(vehicle_type, 3)

                box = target.find('box')
                left = float(box['left'])
                top = float(box['top'])
                width = float(box['width'])
                height = float(box['height'])

                # calculate the center coordinates and normalize them
                x_center = (left + width / 2) / IMAGE_WIDTH # the image width is 960 pixels
                y_center = (top + height / 2) / IMAGE_HEIGHT # the image height is 540 pixels

                # normalize the width and height
                w = width / 960
                h = height / 540

                # prepare data to write to file
                data =  f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

                #print(number, data)

                # use f-strings to format the new file name
                new_file_name = f"img{number:05d}{file_name[:-4]}.txt"
                #print(new_file_name)

                # use with open to write files
                with open(path.join("../temp", loc, file_name, new_file_name), "a") as f:
                    f.write(data)


def relocate_annotations():
    print("Running relocate process..")
    
    print("Relocating training labels")
    for folder in listdir(f"{temp_annotations_loc}/train"):
        print(folder)
        for file in listdir(f"{temp_annotations_loc}/train/{folder}"):
            shutil.move(f"{temp_annotations_loc}/train/{folder}/{file}", f"{str(root_output_dir)}/labels/train/{file}")

    print("Relocating val labels")
    for folder in listdir(f"{temp_annotations_loc}/val"):
        print(folder)
        for file in listdir(f"{temp_annotations_loc}/val/{folder}"):
            shutil.move(f"{temp_annotations_loc}/val/{folder}/{file}", f"{str(root_output_dir)}/labels/val/{file}")

    rmtree("../temp")

    print("--SUCCESS! All annotations files relocated--")

def verify_success():
    total_train_images = len(glob.glob(path.join(root_output_dir, "images", "train", "*")))
    total_val_images = len(glob.glob(path.join(root_output_dir, "images", "val", "*")))
    
    total_train_labels = len(glob.glob(path.join(root_output_dir, "labels", "train", "*")))
    total_val_labels = len(glob.glob(path.join(root_output_dir, "labels", "val", "*")))

    print(f"train images: {total_train_images} |", f"train labels: {total_train_labels};")
    print(f"valuation images: {total_val_images} |", f"valuation labels: {total_val_labels};")

    if not (total_train_images > 0 or total_val_images > 0):
        raise Exception("Failed to relocate either train images/labels!")
    elif not (total_val_images > 0 or total_val_labels > 0):
        raise Exception("Failed to relocate either test images/labels!")
    
    assert total_train_images == 83791, "The number of train images does not match the expected number (expected 56340)"
    assert total_val_images == 56340, "The number of val images does not match the expected number (expected 83791)"
    assert total_train_labels == 82085, "The number of train labels does not match the expected number (expected 56340)"
    assert total_val_labels == 56167, "The number of val labels does not match the expected number (expected 83791)"

def ready_output_dir():
    if path_exists("../temp"):
        remove_if_exists("../temp", message="Found existing temp folder, removing...")

    folders = [
        "../temp/train",
        "../temp/val"
    ]

    make_dirs(folders)

    #Generate train label data structure
    paths = listdir(xml_loc_train)
    for i in paths:
        mkdir("../temp/train/" + i)

    #Generate test label data structure
    paths = listdir(xml_loc_test)
    for i in paths:
        mkdir("../temp/val/" + i)


if __name__ == "__main__":
    ready_output_dir()

    for i in ["train", "val"]:
        xml_to_yolov8_format(i)
    
    print(f"""
            ---DONE! SUCCESSFULLY PRE-PARSED ANNOTATIONS---
                train backgrounds: {train_backgrounds}
                test backgrounds: {test_backgrounds}
            -----------------------------------------------"""
        )

    image_preprocess.run_image_preprocess()
    relocate_annotations()
    verify_success()
