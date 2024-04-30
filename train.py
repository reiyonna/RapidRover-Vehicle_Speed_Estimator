import requests
import argparse
import ultralytics
 
import sys
import os
sys.path.append(os.path.dirname("./"))
import utils.path as pathutil

def init_argparse() -> argparse.Namespace:
    """Initialise the argument parser for the file"""
    parser = argparse.ArgumentParser(
        prog='Train',
        description='Trains yolov8 model')
    parser.add_argument('-v', '--validate', type=bool, default=False)
    parser.add_argument('-p', '--path', type=pathutil.pt_dir_path, default=None)
    args = parser.parse_args()
    return args

def train_data(path):
    """Trains the preprocessed data on yolov8 medium model"""
    if path:
        model = ultralytics.YOLO(path) 
    else:
        if not pathutil.path_exists("./yolov8m.pt"):
            res = input("No path parameter specified (python3 train.py PATH_TO_PT_FILE), would you like to train with yolov8 model instead? Y/N:")
            print(res)
            if res.lower() == "y":
                print("Downloading yolov8m model...")
                requests.get("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt")
            else:
                print("Didn't receive 'Y' as response, exiting...")
                return
        model = ultralytics.YOLO("yolo8m.pt")

    ultralytics.checks()
    model.to('cuda')

    model.train(data="config.yaml", epochs=100, save_period=2)

# testing code
def validate(path):
    if not path:
        raise Exception("A path to the .pt file needs to be provided for validating")
    model = ultralytics.YOLO(path) 
    model.val()

if __name__ == '__main__':
    args = init_argparse()
    if not args.validate:
        train_data(args.path)
    else:
        validate(args.path)