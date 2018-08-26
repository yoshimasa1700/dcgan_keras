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
    files = list(_find_all_files_with_ext("../../../data/lfw", "jpg"))

    for file_name in files:
        image = cv2.imread(file_name)
        cv2.imshow("img", image)
        print(type(image))
        print(image.shape)
        cv2.waitKey(1)
    
    # options = parse_arg()
    # print(options.mode)

    # if options.mode == "train":
    #     trainer = Trainer()
    #     trainer.train()

    # if options.mode == "generate":
    #     generator = Generator()
    #     generator.load_model("generator.h5")
    #     generator.generate()
