import math
from PIL import Image
import numpy as np


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = int(math.ceil(float(total)/cols))

    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image


def save_images(generated_images, path):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5

    Image.fromarray(image.astype(np.uint8)).save(
        path)
