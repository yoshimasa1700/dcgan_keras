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

    
    dataset = np.ndarray((0, 64, 64, 3))

    print(dataset)
    print(dataset.shape)
    
    for file_name in files:
        image = cv2.imread(file_name)

        resized_image = cv2.resize(image[50: 200, 50: 200], (64, 64))
        # cv2.imshow("img", resized_image)

        resized_image = np.reshape(resized_image, (1, 64, 64, 3))
        
        # print(resized_image.shape)

        dataset = np.concatenate((dataset, resized_image), axis=0)
        print(dataset.shape)

    print(dataset.shape)
    options = parse_arg()
    print(options.mode)

    if options.mode == "train":
        trainer = Trainer()
        trainer.train(dataset)

    # if options.mode == "generate":
    #     generator = Generator()
    #     generator.load_model("generator.h5")
    #     generator.generate()
