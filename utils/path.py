import argparse
import os
from shutil import rmtree

def pt_dir_path(path):
    if path[-3:] != ".pt":
        raise argparse.ArgumentTypeError(f"Can only train from .pt files! The path you have provided needs to link to a .pt file! ({path})")
    
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
        
def path_exists(path: str) -> bool:
    return os.path.exists(path)

def make_dirs(arr):
    """Recursively creates paths that doesn't exist, any intermediate path segment 
    (not just the rightmost) will be created if it does not exist.

    Args:
        arr (list): string of paths to create
    """
    for path in arr:
        os.makedirs(path)

def remove_if_exists(path: str, message = "Found existing path, removing..."):
    if os.path.exists(path):
        print(f"{message} ({path})")
        rmtree(path)


