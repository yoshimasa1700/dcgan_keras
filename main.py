import math
import numpy as np
import argparse

from Trainer import Trainer
from Generator import Generator


def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--mode", type=str, default="generate")

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    options = parse_arg()
    print(options.mode)

    if options.mode == "train":
        trainer = Trainer()
        trainer.train()

    if options.mode == "generate":
        generator = Generator()
        generator.load_model("generator.h5")
        generator.generate()
