import math
import numpy as np
import argparse
import os
import cv2

from Trainer import Trainer
from Generator import Generator


def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--mode", type=str, default="generate")

    args = parser.parse_args()
    return args


def _find_all_files_with_ext(dir, ext):
    suffix = os.extsep + ext.lower()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(suffix):
                yield os.path.join(root, file)


if __name__ == "__main__":
    options = parse_arg()

    if options.mode == "train":
        trainer = Trainer()
        trainer.train("/mnt/data2/lsun/kitchen_train_lmdb")

    # if options.mode == "generate":
    #     generator = Generator()
    #     generator.load_model("generator.h5")
    #     generator.generate()
